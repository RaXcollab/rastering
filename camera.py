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


# --- ini load/save/apply + DEFAULT_INI --------------------------------------
# Schema (.ini sections / keys):
#   [Image size]   Width, Height, Start X, Start Y   (absolute top-left)
#   [Timing]       ExposureMs, AcqFrameRate, AcqFrameRateEnable
#   [Analog]       GainDB, Gamma, GammaEnable, PixelFormat,
#                  BlackLevel, BlackLevelClamping, DefectCorrection
#   [GigE]         PacketSize, PacketDelay, ThroughputLimit
#   [Display]      rotation_k, flip_x, flip_y   (UNCHANGED from uEye era;
#                  ui.py:390/498 callers depend on this schema)
#
# Legacy uEye Cockpit keys are also accepted (auto-migration):
#   [Timing]/Pixelclock        -> dropped (Parity 4: GigE has no pixel clock)
#   [Timing]/Exposure          -> exposure_ms (same units)
#   [Timing]/Framerate         -> acq_frame_rate
#   [Timing]/TargetFPS         -> acq_frame_rate
#   [Timing]/TimingMode        -> dropped (Parity 5)
#   [Gain]/Master              -> dropped (Parity 2: int 0-100 has no dB analog)
#   [Gain]/GainBoost           -> dropped (Parity 2)
#   [Parameters]/Gamma         -> [Analog]/Gamma
#   [Parameters]/Hotpixel Mode -> applied via apply_ini_to_camera (decision 8)
#   [Parameters]/Hardware Gamma-> applied via apply_ini_to_camera (decision 8)
# Migration warnings fire once per legacy key per process (via the
# CameraConfig._legacy_warned ClassVar tracking).

DEFAULT_INI: str = """\
# camera_params_spin.ini -- Spinnaker rastering camera defaults.
# Schema documented at GUIs/rastering/camera.py near DEFAULT_INI.
# Legacy uEye Cockpit (.ini) files are auto-migrated by
# load_ueye_config_from_ini() with one-time deprecation warnings.

[Image size]
Width = 0
Height = 0
Start X = 0
Start Y = 0

[Timing]
ExposureMs = 30.0
AcqFrameRate = 20.0
AcqFrameRateEnable = 1

[Analog]
GainDB = 0.0
Gamma = 1.0
GammaEnable = 0
PixelFormat = Mono8
BlackLevel = 0.0
BlackLevelClamping = 1
DefectCorrection = 0

[GigE]
PacketSize = 9000
PacketDelay = 1000
ThroughputLimit =

[Display]
rotation_k = -1
flip_x = 0
flip_y = 1
"""

_TRUE_STR = ("1", "true", "yes", "on")


def _read_cp(ini_path: str):
    """Read an .ini into a ConfigParser. Missing files return an empty parser
    (caller can detect via ``cp.sections()``); read errors log + empty."""
    import configparser
    cp = configparser.ConfigParser()
    if not os.path.exists(ini_path):
        return cp
    try:
        cp.read(ini_path, encoding="utf-8-sig")
    except Exception as e:
        logger.warning("ini read failed for %s: %s", ini_path, e)
    return cp


def _get(cp, section: str, key: str, fallback: Optional[str] = None) -> Optional[str]:
    import configparser
    try:
        return cp.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback


def _getbool_str(s: Optional[str], fallback: bool = False) -> bool:
    if s is None:
        return fallback
    return s.strip().lower() in _TRUE_STR


def load_ueye_config_from_ini(ini_path: str, **overrides: Any) -> CameraConfig:
    """Parse a camera .ini and return a CameraConfig.

    Reads both the new Spinnaker schema and the legacy uEye-Cockpit schema;
    legacy keys are auto-migrated via the ``CameraConfig`` legacy-kwargs
    absorber (one-time deprecation warning per key). New-schema keys win
    over legacy if both are present.

    Missing file -> returns ``CameraConfig(**overrides)`` with defaults.
    Any keyword in ``overrides`` takes final precedence over the file.
    The function name is preserved from the uEye era because ``ui.py:483``
    imports it by that name."""
    if not os.path.exists(ini_path):
        logger.info(
            "camera config .ini not found at %s; using CameraConfig defaults", ini_path
        )
        return CameraConfig(**overrides)

    cp = _read_cp(ini_path)
    kwargs: Dict[str, Any] = {}

    # -- Image size --------------------------------------------------------
    w = _get(cp, "Image size", "Width")
    if w is not None:
        try:
            wv = int(w)
            if wv > 0:
                kwargs["width"] = wv
        except ValueError:
            pass
    h = _get(cp, "Image size", "Height")
    if h is not None:
        try:
            hv = int(h)
            if hv > 0:
                kwargs["height"] = hv
        except ValueError:
            pass
    for ini_key, cfg_key in (("Start X", "roi_offset_x"), ("Start Y", "roi_offset_y")):
        v = _get(cp, "Image size", ini_key)
        if v is not None:
            try:
                iv = int(v)
                if iv >= 0:
                    kwargs[cfg_key] = iv
            except ValueError:
                pass

    # -- Timing (legacy + Spinnaker) ---------------------------------------
    # ExposureMs (new) wins over Exposure (legacy ms units)
    exp = _get(cp, "Timing", "ExposureMs") or _get(cp, "Timing", "Exposure")
    if exp is not None:
        try:
            kwargs["exposure_ms"] = float(exp)
        except ValueError:
            pass

    # AcqFrameRate (new) > TargetFPS (legacy) > Framerate (legacy alternate)
    afr = (
        _get(cp, "Timing", "AcqFrameRate")
        or _get(cp, "Timing", "TargetFPS")
        or _get(cp, "Timing", "Framerate")
    )
    if afr is not None:
        try:
            v = float(afr)
            if v > 0:
                kwargs["acq_frame_rate"] = v
        except ValueError:
            pass

    afre = _get(cp, "Timing", "AcqFrameRateEnable")
    if afre is not None:
        kwargs["acq_frame_rate_enable"] = _getbool_str(afre, True)

    # Legacy uEye keys -> let the absorber drop them with deprecation warnings
    pclk = _get(cp, "Timing", "Pixelclock")
    if pclk is not None:
        try:
            kwargs["pixel_clock_mhz"] = int(pclk)  # absorber drops; warns once
        except ValueError:
            pass
    tm = _get(cp, "Timing", "TimingMode")
    if tm is not None:
        kwargs["prioritize_exposure"] = (tm.strip().lower() == "exposure")  # absorber drops

    # -- Analog (new) + Gain / Parameters (legacy) -------------------------
    gdb = _get(cp, "Analog", "GainDB")
    if gdb is not None:
        try:
            kwargs["gain_db"] = float(gdb)
        except ValueError:
            pass
    else:
        # Legacy [Gain]/Master is int 0-100 with no Spinnaker analog --
        # absorber drops with warning per Parity 2.
        gm = _get(cp, "Gain", "Master")
        if gm is not None:
            try:
                kwargs["master_gain"] = int(gm)
            except ValueError:
                pass
    gboost = _get(cp, "Gain", "GainBoost")
    if gboost is not None:
        kwargs["enable_gain_boost"] = _getbool_str(gboost)  # absorber drops

    g = _get(cp, "Analog", "Gamma") or _get(cp, "Parameters", "Gamma")
    if g is not None:
        try:
            kwargs["gamma"] = float(g)
        except ValueError:
            pass

    ge = _get(cp, "Analog", "GammaEnable")
    if ge is not None:
        kwargs["gamma_enable"] = _getbool_str(ge)

    pf = _get(cp, "Analog", "PixelFormat")
    if pf is not None and pf.strip():
        kwargs["pixel_format"] = pf.strip()

    bl = _get(cp, "Analog", "BlackLevel")
    if bl is not None:
        try:
            kwargs["black_level"] = float(bl)
        except ValueError:
            pass

    blc = _get(cp, "Analog", "BlackLevelClamping")
    if blc is not None:
        kwargs["black_level_clamping"] = _getbool_str(blc, True)

    dc = _get(cp, "Analog", "DefectCorrection")
    if dc is not None:
        kwargs["defect_correction"] = _getbool_str(dc)

    # -- GigE --------------------------------------------------------------
    ps = _get(cp, "GigE", "PacketSize")
    if ps is not None:
        try:
            kwargs["gige_packet_size"] = int(ps)
        except ValueError:
            pass
    pd = _get(cp, "GigE", "PacketDelay")
    if pd is not None:
        try:
            kwargs["gige_packet_delay"] = int(pd)
        except ValueError:
            pass
    tl = _get(cp, "GigE", "ThroughputLimit")
    if tl is not None and tl.strip():
        try:
            kwargs["device_link_throughput_limit"] = int(tl)
        except ValueError:
            pass

    # -- Overrides win over the file ---------------------------------------
    kwargs.update(overrides)
    return CameraConfig(**kwargs)


def save_settings_to_ini(ini_path: str, settings: Dict[str, Any]) -> None:
    """Write the current camera + display settings to an .ini file.

    ``settings`` is the dict produced by
    ``CameraSettingsDock.get_current_settings()`` (Step 4 redesigns the
    schema -- this function tolerates both old uEye keys and new Spinnaker
    keys during the migration window). Display-only keys (``rotation_k``,
    ``flip_x``, ``flip_y``) go into the ``[Display]`` section -- schema
    preserved from the uEye era (ui.py:390/498).

    Reads the existing file first if present, so unrelated sections survive
    (rollback friendliness). The new Spinnaker schema sections are written
    in lockstep with the new dock keys; legacy keys (gain int 0-100,
    pixel_clock, timing_mode) are intentionally dropped -- their Parity
    counterparts are written instead.
    """
    import configparser
    cp = configparser.ConfigParser()
    if os.path.exists(ini_path):
        try:
            cp.read(ini_path, encoding="utf-8-sig")
        except Exception as e:
            logger.warning("save_settings_to_ini: existing %s unreadable: %s", ini_path, e)

    for section in ("Image size", "Timing", "Analog", "GigE", "Display"):
        if not cp.has_section(section):
            cp.add_section(section)

    def _set(section: str, key: str, val: Any) -> None:
        cp.set(section, key, str(val))

    def _set_bool(section: str, key: str, val: Any) -> None:
        cp.set(section, key, "1" if bool(val) else "0")

    # Image size
    if "aoi_width" in settings:
        _set("Image size", "Width", int(settings["aoi_width"]))
    if "aoi_height" in settings:
        _set("Image size", "Height", int(settings["aoi_height"]))
    if "aoi_x" in settings:
        _set("Image size", "Start X", int(settings["aoi_x"]))
    if "aoi_y" in settings:
        _set("Image size", "Start Y", int(settings["aoi_y"]))

    # Timing
    # exposure_ms (new) wins; fall back to legacy 'exposure' (also ms)
    exp = settings.get("exposure_ms", settings.get("exposure"))
    if exp is not None:
        _set("Timing", "ExposureMs", f"{float(exp):.6f}")
    # acq_frame_rate (new) wins over legacy 'target_fps'
    afr = settings.get("acq_frame_rate", settings.get("target_fps"))
    if afr is not None:
        _set("Timing", "AcqFrameRate", f"{float(afr):.6f}")
    if "acq_frame_rate_enable" in settings:
        _set_bool("Timing", "AcqFrameRateEnable", settings["acq_frame_rate_enable"])

    # Analog
    if "gain_db" in settings:
        _set("Analog", "GainDB", f"{float(settings['gain_db']):.6f}")
    if "gamma" in settings:
        _set("Analog", "Gamma", f"{float(settings['gamma']):.6f}")
    if "gamma_enable" in settings:
        _set_bool("Analog", "GammaEnable", settings["gamma_enable"])
    if "pixel_format" in settings:
        _set("Analog", "PixelFormat", str(settings["pixel_format"]))
    if "black_level" in settings:
        _set("Analog", "BlackLevel", f"{float(settings['black_level']):.6f}")
    if "black_level_clamping" in settings:
        _set_bool("Analog", "BlackLevelClamping", settings["black_level_clamping"])
    if "defect_correction" in settings:
        _set_bool("Analog", "DefectCorrection", settings["defect_correction"])

    # GigE
    if "gige_packet_size" in settings or "packet_size" in settings:
        _set("GigE", "PacketSize", int(settings.get("gige_packet_size", settings.get("packet_size", 9000))))
    if "gige_packet_delay" in settings or "packet_delay" in settings:
        _set("GigE", "PacketDelay", int(settings.get("gige_packet_delay", settings.get("packet_delay", 1000))))
    if "device_link_throughput_limit" in settings or "throughput_limit" in settings:
        tl = settings.get("device_link_throughput_limit", settings.get("throughput_limit"))
        cp.set("GigE", "ThroughputLimit", "" if tl is None else str(int(tl)))

    # Display (schema unchanged from uEye era)
    if "rotation_k" in settings:
        _set("Display", "rotation_k", int(settings["rotation_k"]))
    if "flip_x" in settings:
        _set_bool("Display", "flip_x", settings["flip_x"])
    if "flip_y" in settings:
        _set_bool("Display", "flip_y", settings["flip_y"])

    with open(ini_path, "w", encoding="utf-8") as f:
        cp.write(f)


def apply_ini_to_camera(cam: "SpinCamera", ini_path: str) -> None:
    """Apply .ini extras to a (post-open) ``SpinCamera``.

    Per decision 8: map uEye-era "extras" to their Blackfly-native nodes:
        [Parameters]/Hotpixel Mode  > 0    -> DefectCorrectionEnable=True
        [Parameters]/Hardware Gamma > 0    -> GammaEnable=True
        [Analog]/BlackLevel                -> BlackLevel (float)
        [Analog]/BlackLevelClamping        -> BlackLevelClamping (bool)
        [Analog]/DefectCorrection          -> DefectCorrectionEnable (bool)
        [Image size]/(Width, Height,
                       Start X, Start Y)   -> cam.reinit_aoi(...)

    Each call is guarded: a missing/unwritable node on this Blackfly model
    logs and continues, never raises (per the Spinnaker driver's
    ``_set_node`` helper contract, populated in sub-step 2.3). NOT a no-op
    -- the uEye driver's hotpixel/hardware-gamma capability is preserved
    here, just routed through native Spinnaker controls. The real
    underlying calls land when ``SpinCamera.set_*`` arrive in 2.3."""
    if not os.path.exists(ini_path):
        logger.info("apply_ini_to_camera: %s not found; skipping", ini_path)
        return

    cp = _read_cp(ini_path)

    # Legacy uEye extras -> Blackfly-native nodes (decision 8)
    try:
        hp_mode = int(_get(cp, "Parameters", "Hotpixel Mode", "-1") or "-1")
    except ValueError:
        hp_mode = -1
    if hp_mode > 0:
        try:
            cam.set_defect_correction(True)  # type: ignore[attr-defined]
        except Exception as e:
            logger.info("apply_ini_to_camera: DefectCorrection unavailable: %s", e)

    try:
        hw_gamma = int(_get(cp, "Parameters", "Hardware Gamma", "0") or "0")
    except ValueError:
        hw_gamma = 0
    if hw_gamma > 0:
        try:
            cam.set_gamma_enable(True)  # type: ignore[attr-defined]
        except Exception as e:
            logger.info("apply_ini_to_camera: GammaEnable unavailable: %s", e)

    # New Spinnaker keys
    bl = _get(cp, "Analog", "BlackLevel")
    if bl is not None:
        try:
            cam.set_black_level(float(bl))  # type: ignore[attr-defined]
        except Exception as e:
            logger.info("apply_ini_to_camera: BlackLevel unavailable: %s", e)

    blc = _get(cp, "Analog", "BlackLevelClamping")
    if blc is not None:
        try:
            cam.set_black_level_clamping(_getbool_str(blc, True))  # type: ignore[attr-defined]
        except Exception as e:
            logger.info("apply_ini_to_camera: BlackLevelClamping unavailable: %s", e)

    dc = _get(cp, "Analog", "DefectCorrection")
    if dc is not None:
        try:
            cam.set_defect_correction(_getbool_str(dc))  # type: ignore[attr-defined]
        except Exception as e:
            logger.info("apply_ini_to_camera: DefectCorrection unavailable: %s", e)

    # AOI from the .ini's exact start/size
    try:
        ini_sx = int(_get(cp, "Image size", "Start X", "-1") or "-1")
        ini_sy = int(_get(cp, "Image size", "Start Y", "-1") or "-1")
        ini_w = int(_get(cp, "Image size", "Width", "-1") or "-1")
        ini_h = int(_get(cp, "Image size", "Height", "-1") or "-1")
        if ini_sx >= 0 and ini_sy >= 0 and ini_w > 0 and ini_h > 0:
            cam.reinit_aoi(ini_w, ini_h, ini_sx, ini_sy)
    except Exception as e:
        logger.info("apply_ini_to_camera: AOI apply failed: %s", e)


def _load_display_settings_from_ini(ini_path: str) -> Dict[str, Any]:
    """Read ``[Display]`` ``rotation_k`` / ``flip_x`` / ``flip_y`` from an .ini.

    Schema unchanged from the uEye era (ui.py:390/498 callers depend on it).
    Returns an empty dict if the section is missing (e.g. a pure
    uEye-Cockpit export with no Display section)."""
    if not os.path.exists(ini_path):
        return {}
    cp = _read_cp(ini_path)
    out: Dict[str, Any] = {}
    rk = _get(cp, "Display", "rotation_k")
    if rk is not None:
        try:
            out["rotation_k"] = int(rk)
        except ValueError:
            pass
    fx = _get(cp, "Display", "flip_x")
    if fx is not None:
        out["flip_x"] = _getbool_str(fx)
    fy = _get(cp, "Display", "flip_y")
    if fy is not None:
        out["flip_y"] = _getbool_str(fy)
    return out


def seed_default_ini(ini_path: str) -> bool:
    """Write ``DEFAULT_INI`` to ``ini_path`` if the file does not exist.

    Returns True if the file was created, False if it already existed. Use
    from a launcher / config bootstrap to materialize the Spinnaker .ini on
    first run (decision: keep the existing ``camera_params.ini`` untouched
    as a rollback artifact; the new file is ``camera_params_spin.ini``)."""
    if os.path.exists(ini_path):
        return False
    try:
        with open(ini_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_INI)
    except Exception as e:
        logger.warning("seed_default_ini: failed to write %s: %s", ini_path, e)
        return False
    logger.info("seed_default_ini: wrote default config to %s", ini_path)
    return True


# --- Bottom-of-module aliases (decision 3) ----------------------------------
# Backward-compatible names for callers that still import the uEye-era
# symbols. ui.py:28 and ui.py:470 import these; renaming would require a
# bigger same-patch edit, deferred to Step 3.
UEyeConfig = CameraConfig
UEyeCamera = SpinCamera
UEyeCameraThread = CameraThread
