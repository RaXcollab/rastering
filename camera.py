"""Spinnaker (rotpy) camera driver for the rastering GUI.

Replaces the previous IDS uEye driver; the uEye driver is preserved verbatim
at ``camera_ueye.py`` for reference/diff (commit ``d46ea66``).

This file is the new active driver and is grown across sub-step commits
(2.1 = skeleton; 2.2 = ``CameraConfig`` schema; 2.3 = ``SpinCamera`` I/O;
2.4 = ``CameraThread`` slots/signals; 2.5 = ``CameraThread.run()`` loop;
2.6 = ini load/save). See ``~/.claude/plans/i-have-gotten-new-wondrous-shore.md``
"Step 2 - Execution Sequence" for the full plan.

Audit-mandated invariants (preserved across all sub-steps):
    B1  bounded ``get_next_image`` timeout; loop polls ``isInterruptionRequested()``
        >= 2/sec via the timeout-as-None-sentinel pattern.
    B2  ``SpinCamera.open()`` exception-safe (own try/except teardown across init
        steps); ``SpinCamera.close()`` idempotent across partial init.
    S1  ``grab()`` returns ``np.ascontiguousarray`` copy with ``.base is None``,
        BEFORE ``img.release()`` returns the SDK buffer to the pool.
    S2  ``_pending`` snapshot+clear is atomic under ``_params_lock``; apply is
        OUTSIDE the lock (no C++ I/O held under a mutex).
    S3  every ``set_*`` / ``request_*`` slot body is solely the mutex-guarded
        ``_pending`` dict write -- no slot touches ``self._cam`` from the GUI
        thread.
    N1  trailing-int parser on grab-exception messages; only timeout/incomplete
        codes go in ``_TRANSIENT_GRAB_ERROR_CODES``; everything else is immediate.
    N2  grab returns uint8 2-D ndarray for Mono8 (the format the pyqtgraph view
        consumes with ``autoLevels=False``).
"""

# --- Module-level OpenMP workaround (LOAD-BEARING) -------------------------
# MUST be set BEFORE the first rotpy or numpy import. The env has three
# OpenMP DLLs (rotpy's bundled libiomp5md.dll, numpy MKL's libiomp5md.dll,
# conda's libomp.dll); without this the second to initialize aborts the
# process with "OMP: Error #15". See docs/ROTPY_BUILD.md Section 3 and the
# 2026-05-22 debug session (commit 9d3e91c) for the canonical analysis.
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import time
from dataclasses import dataclass, field, fields as _dc_fields, replace as _dc_replace
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5 import QtCore

logger = logging.getLogger(__name__)


# --- Transient error codes (refined in sub-step 2.5) -----------------------
# Initial empty set per AUDIT:N1 safe default: no trailing-int code in
# rotpy ``SpinnakerAPIException`` messages -> not transient -> immediate emit.
# Note that timeout + incomplete-image are handled BEFORE this codepath via
# the None-sentinel pattern in ``SpinCamera.grab()`` (sub-step 2.3) -- only
# genuine SDK exceptions reach the throttle. Refine if hardware testing
# surfaces a class of recoverable exceptions worth throttling.
_TRANSIENT_GRAB_ERROR_CODES: frozenset = frozenset()
_ERROR_THROTTLE_WINDOW_S: float = 5.0


def _ensure_spinnaker_runtime() -> None:
    """No-op. rotpy's ``__init__.py`` owns DLL discovery via its bundled
    ``share/rotpy/spinnaker/{bin,cti}``. Prepending the installed Teledyne
    Spinnaker SDK (currently 4.3.0.190) onto PATH or via
    ``os.add_dll_directory`` would shadow rotpy's bundled v2.6 runtime and
    crash the bindings at first node call. The function name is retained
    so legacy callers that reference it don't ``AttributeError``."""
    return None


# --- CameraConfig -----------------------------------------------------------
# Legacy-kwargs map: keys that the old uEye driver accepted but the Spinnaker
# driver does not need. Two policies:
#   - DROPS: silently dropped (with a one-time deprecation warning). The
#            corresponding behavior is replaced by a different Spinnaker
#            concept (e.g. pixel_clock -> DeviceLinkThroughputLimit, set via
#            ``device_link_throughput_limit``).
#   - RENAMES: value-preserving rename to a new field name.
# See plan "Parity & Behavioral Differences" sections 2, 4, 5 for the
# physics-side rationale on each drop.
_LEGACY_KWARG_DROPS: frozenset = frozenset({
    "pixel_clock_mhz",        # Parity 4: GigE has no pixel clock
    "master_gain",            # Parity 2: int 0-100 has no Spinnaker analog
    "enable_gain_boost",      # Parity 2: no Spinnaker analog
    "use_freeze",             # Parity 1: replaced by NewestOnly buffer-handling
    "prioritize_exposure",    # Parity 5: replaced by AcquisitionFrameRateEnable
})
_LEGACY_KWARG_RENAMES: Dict[str, str] = {
    "target_fps": "acq_frame_rate",  # value preserved; same semantic meaning
}


@dataclass(init=False)
class CameraConfig:
    """Spinnaker camera config.

    Field schema is Spinnaker-native (dB gain, ms exposure, NewestOnly buffer,
    GigE knobs, decision-8 Blackfly-native nodes). Legacy uEye kwargs
    (``pixel_clock_mhz`` / ``master_gain`` / ``enable_gain_boost`` /
    ``use_freeze`` / ``prioritize_exposure``) are absorbed via ``__init__``
    with a one-time per-key deprecation warning. ``target_fps`` is the only
    legacy kwarg that's value-preservingly RENAMED to ``acq_frame_rate``
    (same semantic, ``config.py:77`` feeds it through ``ui.py:540`` ctor per
    AUDIT:N4)."""

    # Selection (decision 7: single camera, prefer serial; camera_id fallback
    # for backward compat). If serial is set, it wins over camera_id.
    serial: Optional[str] = None
    camera_id: int = 0

    # Geometry: 0 = full-sensor for w/h. ROI offsets are absolute top-left
    # (Spinnaker convention; Parity 6).
    width: int = 0
    height: int = 0
    roi_offset_x: int = 0
    roi_offset_y: int = 0

    # Acquisition
    exposure_ms: float = 30.0
    gain_db: float = 0.0
    gamma: float = 1.0
    gamma_enable: bool = False
    pixel_format: str = "Mono8"
    acq_frame_rate: float = 20.0
    acq_frame_rate_enable: bool = True

    # GigE transport (Parity 4 replaces uEye pixel-clock)
    device_link_throughput_limit: Optional[int] = None  # None = leave at camera default
    gige_packet_size: int = 9000
    gige_packet_delay: int = 1000

    # TLStream buffer policy. "NewestOnly" replaces uEye FreezeVideo's
    # implicit fresh-frame guarantee (Parity 1, AUDIT:parity-1). Without this
    # the continuous-acquisition stream lags the camera under any GUI hiccup.
    buffer_handling: str = "NewestOnly"

    # Blackfly-native nice-to-haves (decision 8). Each is guarded at the
    # SpinCamera layer -- a Blackfly model that lacks the node logs and
    # continues, never raises.
    black_level: float = 0.0
    black_level_clamping: bool = True
    defect_correction: bool = False

    # Display
    emit_rgb: bool = False  # if True, grab() returns (H,W,3); else (H,W)

    # Soft cap on the run-loop frame rate (independent of the camera's own
    # AcquisitionFrameRate; protects the GUI thread).
    max_fps: float = 30.0

    # Module-level set of legacy kwargs we've already warned about (one-time
    # per process). Tracked on the class so subclasses share the state.
    _legacy_warned: ClassVar[Set[str]] = set()

    def __init__(self, **kwargs: Any) -> None:
        # 1. Pull legacy kwargs out of the user-provided mapping with
        #    one-time deprecation warnings.
        cleaned: Dict[str, Any] = {}
        for key, val in kwargs.items():
            if key in _LEGACY_KWARG_DROPS:
                self._warn_once_legacy(key, action="dropped (Spinnaker has no analog)")
                continue
            if key in _LEGACY_KWARG_RENAMES:
                target = _LEGACY_KWARG_RENAMES[key]
                if target in kwargs:
                    # User passed both legacy + new name -- new name wins,
                    # legacy is ignored.
                    self._warn_once_legacy(
                        key, action=f"superseded by explicit {target!r}; ignoring legacy"
                    )
                else:
                    self._warn_once_legacy(key, action=f"renamed to {target!r}")
                    cleaned[target] = val
                continue
            cleaned[key] = val

        # 2. Resolve every declared field: pull from cleaned, else default.
        declared = {f.name for f in _dc_fields(self.__class__)}
        for f in _dc_fields(self.__class__):
            setattr(self, f.name, cleaned.pop(f.name, f.default))

        # 3. Unknown leftover kwargs are a hard error (caller bug).
        if cleaned:
            unknown = ", ".join(sorted(cleaned.keys()))
            raise TypeError(
                f"CameraConfig: unknown kwarg(s): {unknown}. "
                f"Known fields: {sorted(declared)}; legacy-accepted: "
                f"{sorted(_LEGACY_KWARG_DROPS | set(_LEGACY_KWARG_RENAMES))}."
            )

    @classmethod
    def _warn_once_legacy(cls, key: str, *, action: str) -> None:
        if key not in cls._legacy_warned:
            cls._legacy_warned.add(key)
            logger.warning("CameraConfig: legacy kwarg %r %s", key, action)


# --- SpinCamera (real I/O in sub-step 2.3) ----------------------------------
class SpinCamera:
    """Spinnaker camera wrapper. Real I/O via rotpy is populated in 2.3."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._opened: bool = False
        self._init_cam_called: bool = False
        self._acquiring: bool = False
        self._cam = None  # rotpy Camera handle, populated in open()
        self._system = None  # rotpy SpinSystem handle, populated in open()

    def open(self) -> None:
        raise NotImplementedError("sub-step 2.3")

    def close(self) -> None:
        """Idempotent across partial init (AUDIT:B2). 2.3 fills the real teardown."""
        return None

    def grab(self) -> Optional[np.ndarray]:
        """Returns uint8 2-D ndarray (Mono8) per AUDIT:N2. None on timeout /
        incomplete (the interruptibility-poll sentinel per AUDIT:B1)."""
        raise NotImplementedError("sub-step 2.3")

    def reinit_aoi(self, width: int, height: int, start_x: int, start_y: int) -> None:
        raise NotImplementedError("sub-step 2.3")

    def get_camera_info(self) -> Dict[str, Any]:
        raise NotImplementedError("sub-step 2.3")


# --- CameraThread (slots in 2.4; run loop in 2.5) ---------------------------
class CameraThread(QtCore.QThread):
    """Camera worker thread.

    Signal contract preserved from the uEye driver (caller-visible API):
        new_frame(np.ndarray)        - emitted on each successful frame
        status(str)                  - lifecycle / informational
        error(str)                   - errors (throttled per AUDIT:N1)
        camera_info_signal(dict)     - ranges + current values, on open / refresh

    Slot contract (AUDIT:S2 + S3): every ``set_*`` / ``request_*`` body is
    solely the mutex-guarded ``_pending`` write -- slots never touch
    ``self._cam`` from the GUI thread. ``run()`` snapshots + clears
    ``_pending`` atomically under the lock, then applies OUTSIDE the lock."""

    new_frame = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    camera_info_signal = QtCore.pyqtSignal(dict)

    def __init__(self, cfg: Optional[CameraConfig] = None, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg if cfg is not None else CameraConfig()
        self._cam: Optional[SpinCamera] = None
        self._running: bool = False
        self._params_lock = QtCore.QMutex()
        self._pending: Dict[str, Any] = {}
        self._err_throttle_state: Dict[str, Tuple[int, float, float]] = {}

    # Track once-per-process alias warnings (distinct from CameraConfig's
    # set; this one fires on slot-invocation, not config construction).
    _alias_warned: ClassVar[Set[str]] = set()

    def stop(self) -> None:
        """Set the loop-exit flag and request interruption. Loop
        interruptibility depends on the bounded ``get_next_image`` timeout
        in ``SpinCamera.grab()`` (AUDIT:B1) -- the timeout returns ``None``
        which causes the loop to re-check ``isInterruptionRequested()`` at
        >= 2/sec."""
        self._running = False
        self.requestInterruption()

    # --- slot helpers ------------------------------------------------------
    def _set_pending(self, key: str, value: Any) -> None:
        """The ONLY thing slots are allowed to do. Mutex-guarded single-key
        write to ``_pending``. See AUDIT:S2 + S3."""
        with QtCore.QMutexLocker(self._params_lock):
            self._pending[key] = value

    def _warn_alias_once(self, slot_name: str, *, action: str) -> None:
        if slot_name not in self._alias_warned:
            self._alias_warned.add(slot_name)
            logger.warning("CameraThread: legacy slot %r %s", slot_name, action)

    # --- Spinnaker-native slots (active in run loop sub-step 2.5) ---------
    @QtCore.pyqtSlot(float)
    def set_exposure_ms(self, v: float) -> None:
        self._set_pending("exposure_ms", float(v))

    @QtCore.pyqtSlot(float)
    def set_gain_db(self, v: float) -> None:
        self._set_pending("gain_db", float(v))

    @QtCore.pyqtSlot(float)
    def set_gamma(self, v: float) -> None:
        self._set_pending("gamma", float(v))

    @QtCore.pyqtSlot(bool)
    def set_gamma_enable(self, v: bool) -> None:
        self._set_pending("gamma_enable", bool(v))

    @QtCore.pyqtSlot(str)
    def set_pixel_format(self, v: str) -> None:
        self._set_pending("pixel_format", str(v))

    @QtCore.pyqtSlot(float)
    def set_acquisition_frame_rate(self, v: float) -> None:
        self._set_pending("acq_frame_rate", float(v))

    @QtCore.pyqtSlot(bool)
    def set_frame_rate_enable(self, v: bool) -> None:
        self._set_pending("acq_frame_rate_enable", bool(v))

    @QtCore.pyqtSlot(int)
    def set_packet_size(self, v: int) -> None:
        self._set_pending("gige_packet_size", int(v))

    @QtCore.pyqtSlot(int)
    def set_throughput_limit(self, v: int) -> None:
        self._set_pending("device_link_throughput_limit", int(v))

    # Decision-8 Blackfly-native (nice-to-have; SpinCamera guards each node
    # so an unavailable Blackfly model degrades silently).
    @QtCore.pyqtSlot(float)
    def set_black_level(self, v: float) -> None:
        self._set_pending("black_level", float(v))

    @QtCore.pyqtSlot(bool)
    def set_black_level_clamping(self, v: bool) -> None:
        self._set_pending("black_level_clamping", bool(v))

    @QtCore.pyqtSlot(bool)
    def set_defect_correction(self, v: bool) -> None:
        self._set_pending("defect_correction", bool(v))

    # --- REAL forwarder (NOT a no-op alias) per AUDIT:S3-domain ----------
    # The dock's _bind_param_controls("set_target_fps") resolves the slot by
    # string name; renaming this would silently kill the FPS slider. Keep
    # it as a real slot that forwards to acq_frame_rate.
    @QtCore.pyqtSlot(float)
    def set_target_fps(self, v: float) -> None:
        self._set_pending("acq_frame_rate", float(v))

    # --- uEye-era alias slots (AUDIT:S4) ---------------------------------
    # These exist so dock callers from the pre-redesign era don't crash; each
    # writes a key into ``_pending`` that the run loop intentionally ignores
    # (set_pixel_clock, set_gain_boost, set_prioritize_exposure) -- the
    # underlying Spinnaker concept doesn't exist (Parity 2/4/5). Aliases
    # MUST exist before any caller fires (Step 4 dock-redesign will delete
    # the call sites; until then these absorb stray events).
    @QtCore.pyqtSlot(int)
    def set_master_gain(self, v: int) -> None:
        """Legacy uEye 0-100 gain. Forwards to set_gain_db (Spinnaker dB);
        note Parity 2 -- the int 0-100 has no physical analog in dB, so the
        value is passed through but the resulting dB may be out-of-range
        and clamped by the camera. New code should call set_gain_db."""
        self._warn_alias_once("set_master_gain", action="forwarded to set_gain_db (Parity 2)")
        self._set_pending("gain_db", float(v))

    @QtCore.pyqtSlot(int)
    def set_pixel_clock(self, v: int) -> None:
        """uEye pixel-clock has no GigE Spinnaker analog (Parity 4). Recorded
        in ``_pending`` for diagnostic visibility but the run loop ignores
        this key. New code should use ``set_throughput_limit`` /
        ``set_packet_size``."""
        self._warn_alias_once(
            "set_pixel_clock",
            action="no-op (Parity 4: GigE has no pixel clock; use throughput_limit instead)",
        )
        self._set_pending("pixel_clock", int(v))

    @QtCore.pyqtSlot(bool)
    def set_gain_boost(self, v: bool) -> None:
        """uEye gain-boost has no Spinnaker analog (Parity 2). No-op alias."""
        self._warn_alias_once("set_gain_boost", action="no-op (Parity 2: no Spinnaker analog)")
        self._set_pending("gain_boost", bool(v))

    @QtCore.pyqtSlot(bool)
    def set_prioritize_exposure(self, v: bool) -> None:
        """uEye prioritize-exposure timing-mode has no Spinnaker analog
        (Parity 5). The Blackfly-native equivalent is unchecking
        ``acq_frame_rate_enable``. No-op alias."""
        self._warn_alias_once(
            "set_prioritize_exposure",
            action="no-op (Parity 5: use acq_frame_rate_enable=False for long exposures)",
        )
        self._set_pending("prioritize_exposure", bool(v))

    # --- request_* (plain methods, no @pyqtSlot) -------------------------
    # ui.py:1325 calls these directly. Same mutex-dict-write contract as
    # @pyqtSlot methods per AUDIT:S2.
    def request_aoi_change(self, width: int, height: int, start_x: int, start_y: int) -> None:
        self._set_pending("aoi", (int(width), int(height), int(start_x), int(start_y)))

    def request_info_refresh(self) -> None:
        self._set_pending("refresh_info", True)

    def request_ini_extras(self, ini_path: str) -> None:
        self._set_pending("ini_extras", str(ini_path))

    def run(self) -> None:
        raise NotImplementedError("sub-step 2.5")


# --- ini load/save (sub-step 2.6) -------------------------------------------
def load_ueye_config_from_ini(ini_path: str, **overrides: Any) -> CameraConfig:
    raise NotImplementedError("sub-step 2.6")


def save_settings_to_ini(ini_path: str, settings: Dict[str, Any]) -> None:
    raise NotImplementedError("sub-step 2.6")


def apply_ini_to_camera(cam: SpinCamera, ini_path: str) -> None:
    raise NotImplementedError("sub-step 2.6")


def _load_display_settings_from_ini(ini_path: str) -> Dict[str, Any]:
    raise NotImplementedError("sub-step 2.6")


# --- Bottom-of-module aliases (decision 3) ----------------------------------
# Backward-compatible names for callers that still import the uEye-era
# symbols. ui.py:28 and ui.py:470 import these; renaming would require a
# bigger same-patch edit, deferred to Step 3.
UEyeConfig = CameraConfig
UEyeCamera = SpinCamera
UEyeCameraThread = CameraThread
