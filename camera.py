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
    S1  ``grab()`` returns ``np.frombuffer(bytearray).reshape(h,w).copy()`` with
        ``.base is None``, BEFORE ``img.release()`` returns the SDK buffer to
        the pool. The intermediate bytearray IS already an owned copy of the
        SDK buffer at the C++ ``ImagePtr`` layer, but the trailing ``.copy()``
        ensures the emitted ndarray references no intermediate at all.
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
import traceback
from dataclasses import dataclass, field, fields as _dc_fields, replace as _dc_replace
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

# IMPORTANT (LOAD-BEARING): C-extension import order on Windows is
#   rotpy -> rotpy.system + rotpy.camera -> numpy -> PyQt5
# in that exact sequence. Verified 2026-05-22 via systematic-debugging:
#   * Smoke (rotpy.system -> numpy, no PyQt5) works in <1 s.
#   * Gate (numpy -> camera-module -> lazy rotpy.system) hangs indefinitely
#     in the Windows DLL loader at ``from rotpy.system import SpinSystem``.
#   * Gate (numpy -> camera-module with rotpy module-top -> PyQt5) also
#     hangs at the eager rotpy.system import.
#   * Gate (camera-module-with-rotpy-FIRST -> numpy -> PyQt5) works.
#
# Mechanism (from Microsoft DLL search docs + rotpy/__init__.py):
#   rotpy's __init__ calls ``os.add_dll_directory(.../spinnaker/bin)`` and
#   discards the returned handle. The DLL search path stays valid only
#   while that handle is alive; once any module top-level statement
#   completes, the handle is GC-eligible. If numpy / PyQt5 load DLLs that
#   modify the loader state before the rotpy .pyd extensions resolve their
#   Spinnaker imports, the bindings either deadlock (no error) or fail
#   with ``DLL load failed -- The operating system cannot run %1``.
#
# Mitigation: load ``rotpy`` and immediately resolve its two
# .pyd-backed submodules in the same import sequence (the handle is still
# on the C stack at that point). After this, ``SpinCamera.open()`` can
# import the same names lazily -- they're cached in ``sys.modules``.
try:
    import rotpy  # noqa: F401  -- side effect: register DLL paths
    from rotpy import system as _rotpy_system  # noqa: F401  -- eager .pyd load
    from rotpy import camera as _rotpy_camera  # noqa: F401  -- eager .pyd load
except ImportError:  # pragma: no cover -- production envs always have rotpy
    rotpy = None  # type: ignore[assignment]

import numpy as np
from PyQt5 import QtCore

logger = logging.getLogger(__name__)


# --- Grab error taxonomy ---------------------------------------------------
# rotpy 0.2.1 ``SpinnakerAPIException.spin_error_code`` values (from
# ``rotpy.names.spin.error_code_names`` -- queryable at runtime, 42 entries).
# The ones we care about in the grab path:
#   -2007  timeout         -- expected on poll; converted to None inside grab()
#   -1010  io              -- transient GigE network jitter / dropped packet
#   -1022  busy            -- transient: camera momentarily handling another op
# Hard (loop-exiting) errors NOT in either set:
#   -1002  not_initialized
#   -1004  resource_in_use   (some other process took the camera)
#   -1005  access_denied
#   -1006  invalid_handle    (camera disconnected mid-stream)
#   -1012  abort
# grab() converts ONLY the timeout to a None sentinel (AUDIT:B1
# interruptibility-poll path); every other SDK exception escapes to run(),
# which then decides:
#   * code in _TRANSIENT_GRAB_ERROR_CODES -> throttled emit + continue loop
#   * code NOT in the set                 -> emit + re-raise into outer
#                                            finally (close + exit thread)
# The transient set starts empty per AUDIT:N1 safe default ("hard errors
# bubble; populate only after observing recoverable codes at V8 stress
# testing"). The grab-side timeout filter is what keeps the loop polling
# isInterruptionRequested() at >=2 Hz.
_GRAB_TIMEOUT_CODES: frozenset = frozenset({-2007})
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

    # --- legacy attribute READ compat (AUDIT:N4 + Step-3 migration window) -
    # The plan accepts and drops legacy kwargs via the absorber above, but the
    # unmodified Step-1 ui.py still READS several of them (e.g.
    # ``cfg.prioritize_exposure`` at ui.py:559/L416, ``cfg.target_fps`` at
    # ui.py:419/562). Per AUDIT:N4 ``target_fps`` requires a read-alias to
    # ``acq_frame_rate``; the others return sensible no-op defaults so the
    # GUI launches against the new schema without crashing. Step 3
    # (ui.py + config.py redesign) deletes all six call sites.
    #
    # IMPLEMENTATION (revised after systematic-debugging Phase 1 audit):
    # Previously these were a ``__getattr__`` fallback that made
    # ``hasattr(cfg, "use_freeze")`` LIE (return True for an attribute that
    # isn't really set anywhere). Verified post-hoc that no real call site
    # uses hasattr/getattr on these names today, but a future caller would
    # get fooled. ``@property`` descriptors are strictly better here:
    #   * hasattr() honest (True iff the property is defined)
    #   * dir(cfg) includes them (introspection sees them)
    #   * vars(cfg) excludes them (they're class attrs, not instance __dict__)
    #   * Read-only by enforcement (no silent setter no-ops)
    #   * dataclasses.fields() / dataclasses.replace() unaffected
    #   * Real typos still raise AttributeError
    # Each property is a single line; the deprecation log fires once at
    # CameraConfig construction time via the absorber, not on every read.

    @property
    def target_fps(self) -> float:
        """Legacy alias for ``acq_frame_rate`` (AUDIT:N4)."""
        return float(self.acq_frame_rate)

    @property
    def master_gain(self) -> int:
        """Legacy uEye int gain. Derived from ``gain_db`` so a .ini reload
        round-trip (ui.py:426 reads master_gain -> set_master_gain ->
        set_gain_db) doesn't clobber the value to 0. <1 dB precision loss
        on round-trip is well below sensor noise."""
        return int(round(self.gain_db))

    @property
    def pixel_clock_mhz(self) -> int:
        """Parity 4: GigE has no pixel clock."""
        return 0

    @property
    def enable_gain_boost(self) -> bool:
        """Parity 2: no Spinnaker analog."""
        return False

    @property
    def use_freeze(self) -> bool:
        """Parity 1: replaced by TLStream NewestOnly buffer handling."""
        return False

    @property
    def prioritize_exposure(self) -> bool:
        """Parity 5: replaced by ``acq_frame_rate_enable=False`` for the
        long-exposure path."""
        return False


# --- SpinCamera (real I/O via rotpy 0.2.1) ---------------------------------
# Node-access pattern (verified against `docs/ROTPY_API.md`, generated by
# runtime introspection 2026-05-22):
#   * Pre-listed nodes live on three convenience attributes of Camera:
#       cam.camera_nodes      -- device-side nodes (PixelFormat, ExposureTime,
#                                Gain, Width/Height/OffsetX/OffsetY, GevSCP*,
#                                AcquisitionMode/Frame*, GammaEnable, Black*,
#                                DefectCorrectionEnable, DeviceSerialNumber, ...)
#       cam.tl_stream_nodes   -- transport-layer stream nodes
#                                (StreamBufferHandlingMode = NewestOnly per
#                                AUDIT:parity-1)
#       cam.tl_dev_nodes      -- transport-layer device nodes
#   * Each node is a typed Spin*Node (Bool/Float/Int/Enum/Str).
#       - Bool/Int/Float/Str: get_node_value()/set_node_value()
#       - Enum: set_node_value_from_str("Name") / get_node_value_as_str()
#       - All carry is_available()/is_writable()/is_readable() from SpinBaseNode
#       - Int/Float carry get_min_value()/get_max_value()/get_increment()
#       - Enum carries get_entries() -> [SpinEnumItemNode] (use .get_enum_name())
class SpinCamera:
    """Spinnaker camera wrapper. See module docstring for AUDIT invariants."""

    # Map external (CameraConfig / API) names to GenICam node names. Kept
    # together so changes ripple in one place.
    _NODES = {
        "pixel_format":   ("camera_nodes", "PixelFormat",                "enum"),
        "exposure_us":    ("camera_nodes", "ExposureTime",               "float"),
        "exposure_auto":  ("camera_nodes", "ExposureAuto",               "enum"),
        "gain_db":        ("camera_nodes", "Gain",                       "float"),
        "gain_auto":      ("camera_nodes", "GainAuto",                   "enum"),
        "gamma":          ("camera_nodes", "Gamma",                      "float"),
        "gamma_enable":   ("camera_nodes", "GammaEnable",                "bool"),
        "width":          ("camera_nodes", "Width",                      "int"),
        "height":         ("camera_nodes", "Height",                     "int"),
        "offset_x":       ("camera_nodes", "OffsetX",                    "int"),
        "offset_y":       ("camera_nodes", "OffsetY",                    "int"),
        "sensor_width":   ("camera_nodes", "SensorWidth",                "int"),
        "sensor_height":  ("camera_nodes", "SensorHeight",               "int"),
        "max_width":      ("camera_nodes", "WidthMax",                   "int"),
        "max_height":     ("camera_nodes", "HeightMax",                  "int"),
        "acq_mode":       ("camera_nodes", "AcquisitionMode",            "enum"),
        "acq_fps":        ("camera_nodes", "AcquisitionFrameRate",       "float"),
        "acq_fps_enable": ("camera_nodes", "AcquisitionFrameRateEnable", "bool"),
        "packet_size":    ("camera_nodes", "GevSCPSPacketSize",          "int"),
        "packet_delay":   ("camera_nodes", "GevSCPD",                    "int"),
        "throughput":     ("camera_nodes", "DeviceLinkThroughputLimit",  "int"),
        "black_level":    ("camera_nodes", "BlackLevel",                 "float"),
        "black_clamp":    ("camera_nodes", "BlackLevelClampingEnable",   "bool"),
        "defect_corr":    ("camera_nodes", "DefectCorrectionEnable",     "bool"),
        "device_model":   ("camera_nodes", "DeviceModelName",            "str"),
        "device_serial":  ("camera_nodes", "DeviceSerialNumber",         "str"),
        "stream_buffer":  ("tl_stream_nodes", "StreamBufferHandlingMode","enum"),
    }

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._opened: bool = False
        self._init_cam_called: bool = False
        self._acquiring: bool = False
        self._cam = None  # rotpy.camera.Camera, populated in open()
        self._system = None  # rotpy.system.SpinSystem, populated in open()
        self._SpinnakerAPIException = None  # captured after lazy import

    # --- node helpers ------------------------------------------------------
    def _resolve(self, alias: str):
        spec = self._NODES.get(alias)
        if spec is None:
            return None, None
        group, name, kind = spec
        container = getattr(self._cam, group, None)
        if container is None:
            return None, kind
        node = getattr(container, name, None)
        return node, kind

    def _node_available(self, alias: str) -> bool:
        node, _ = self._resolve(alias)
        if node is None:
            return False
        try:
            return bool(node.is_available())
        except Exception:
            return False

    def _set_node(self, alias: str, value: Any, *, clamp: bool = False) -> bool:
        """Best-effort node write. Returns True if applied; False if skipped
        (unavailable / unwritable / value rejected). All paths are logged at
        ``INFO`` (skip) or ``WARNING`` (rejected); never raises."""
        node, kind = self._resolve(alias)
        if node is None:
            logger.info("SpinCamera: node %r not present", alias)
            return False
        try:
            if not node.is_available():
                logger.info("SpinCamera: node %r unavailable on this model", alias)
                return False
            if not node.is_writable():
                logger.info("SpinCamera: node %r not writable", alias)
                return False
            if kind == "enum":
                node.set_node_value_from_str(str(value))
            elif kind == "bool":
                node.set_node_value(bool(value))
            elif kind == "int":
                v = int(value)
                if clamp:
                    try:
                        mn, mx = node.get_min_value(), node.get_max_value()
                        try:
                            inc = node.get_increment()
                        except Exception:
                            inc = 0
                        if inc and inc > 0:
                            # Review N-4: round-NEAREST instead of round-DOWN
                            # so requested values land on the closest valid
                            # increment (e.g. Width=13 with inc=8 -> 16, not 8).
                            v = mn + int(round((v - mn) / inc)) * inc
                        v = max(mn, min(mx, v))
                    except Exception:
                        pass
                node.set_node_value(v)
            elif kind == "float":
                v = float(value)
                if clamp:
                    try:
                        mn, mx = node.get_min_value(), node.get_max_value()
                        v = max(mn, min(mx, v))
                    except Exception:
                        pass
                node.set_node_value(v)
            elif kind == "str":
                node.set_node_value(str(value))
            else:
                logger.warning("SpinCamera: unknown node kind %r for alias %r", kind, alias)
                return False
            return True
        except Exception as e:
            logger.warning("SpinCamera: set %r=%r rejected: %s", alias, value, e)
            return False

    def _get_node(self, alias: str, default: Any = None) -> Any:
        node, kind = self._resolve(alias)
        if node is None:
            return default
        try:
            if not node.is_available() or not node.is_readable():
                return default
            if kind == "enum":
                return node.get_node_value_as_str()
            return node.get_node_value()
        except Exception:
            return default

    def _get_range(self, alias: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Return (min, max, inc) for an int/float node, or (None, None, None)."""
        node, kind = self._resolve(alias)
        if node is None or kind not in ("int", "float"):
            return (None, None, None)
        try:
            if not node.is_available() or not node.is_readable():
                return (None, None, None)
            mn = node.get_min_value()
            mx = node.get_max_value()
            try:
                inc = node.get_increment()
            except Exception:
                inc = None
            return (mn, mx, inc)
        except Exception:
            return (None, None, None)

    def _get_enum_entries(self, alias: str) -> List[str]:
        node, kind = self._resolve(alias)
        if node is None or kind != "enum":
            return []
        try:
            if not node.is_available() or not node.is_readable():
                return []
            return [e.get_enum_name() for e in node.get_entries()]
        except Exception:
            return []

    # --- lifecycle --------------------------------------------------------
    def open(self) -> None:
        """Acquire system + camera and bring it to a streaming-ready state.

        Exception-safe per AUDIT:B2 -- any failure after this point unwinds
        through ``_teardown_partial`` before re-raising, so a partial open
        never leaves a claimed camera. The order matters:
            1. SpinSystem + CameraList + camera selection
            2. init_cam (gates node access)
            3. AcquisitionMode = Continuous
            4. TLStream StreamBufferHandlingMode = NewestOnly (parity-1)
            5. GigE knobs (packet size / delay / throughput)
            6. Geometry: PixelFormat -> Width/Height -> OffsetX/OffsetY
            7. ExposureAuto/GainAuto = Off
            8. GammaEnable -> Gamma
            9. AcquisitionFrameRateEnable -> AcquisitionFrameRate
           10. ExposureTime, Gain
           11. Blackfly-native (decision 8): BlackLevelClamping/BlackLevel,
               DefectCorrectionEnable
           12. begin_acquisition
        """
        try:
            from rotpy.system import SpinSystem, SpinnakerAPIException
            from rotpy.camera import CameraList
        except ImportError as e:
            raise RuntimeError(f"rotpy not importable: {e}") from e
        self._SpinnakerAPIException = SpinnakerAPIException

        try:
            logger.info("SpinCamera.open: [1] SpinSystem()")
            self._system = SpinSystem()
            logger.info("SpinCamera.open: [1] OK; enumerating cameras")
            cams = CameraList.create_from_system(
                self._system, update_interfaces=True, update_cams=True
            )
            n = cams.get_size()
            logger.info("SpinCamera.open: [1] %d camera(s) enumerated", n)
            if n == 0:
                raise RuntimeError("No Spinnaker cameras enumerated (check GigE link, IP, jumbo frames).")

            if self.cfg.serial:
                try:
                    self._cam = cams.create_camera_by_serial(str(self.cfg.serial))
                except SpinnakerAPIException as e:
                    raise RuntimeError(
                        f"camera with serial {self.cfg.serial!r} not found "
                        f"(detected {n} cameras): {e}"
                    ) from e
            else:
                self._cam = cams.create_camera_by_index(int(self.cfg.camera_id) if self.cfg.camera_id < n else 0)
                logger.info("SpinCamera.open: cfg.serial unset; bound to camera index 0")

            logger.info("SpinCamera.open: [2] init_cam()")
            self._cam.init_cam()
            self._init_cam_called = True

            logger.info("SpinCamera.open: [3] AcquisitionMode=Continuous")
            self._set_node("acq_mode", "Continuous")

            # Review B-3: on some Blackfly firmware ``StreamBufferHandlingMode``
            # is unavailable until ``begin_acquisition`` selects the stream.
            # Try here first (works on most models); if it returns False the
            # post-begin retry at step [12+] picks it up. AUDIT:parity-1
            # silently no-ops if BOTH attempts fail -- live view will lag.
            logger.info("SpinCamera.open: [4] StreamBufferHandlingMode=%s",
                        self.cfg.buffer_handling)
            stream_buffer_set_ok = self._set_node(
                "stream_buffer", self.cfg.buffer_handling
            )
            if not stream_buffer_set_ok:
                logger.warning(
                    "SpinCamera.open: [4] stream_buffer pre-begin set returned False; "
                    "will retry post begin_acquisition."
                )

            logger.info("SpinCamera.open: [5] GigE: pkt_size=%s pkt_delay=%s tput=%s",
                        self.cfg.gige_packet_size, self.cfg.gige_packet_delay,
                        self.cfg.device_link_throughput_limit)
            if self.cfg.gige_packet_size:
                self._set_node("packet_size", self.cfg.gige_packet_size, clamp=True)
            if self.cfg.gige_packet_delay:
                self._set_node("packet_delay", self.cfg.gige_packet_delay, clamp=True)
            if self.cfg.device_link_throughput_limit is not None:
                self._set_node(
                    "throughput", self.cfg.device_link_throughput_limit, clamp=True
                )

            logger.info("SpinCamera.open: [6] geometry pix_fmt=%s w=%s h=%s ox=%s oy=%s",
                        self.cfg.pixel_format, self.cfg.width, self.cfg.height,
                        self.cfg.roi_offset_x, self.cfg.roi_offset_y)
            self._set_node("pixel_format", self.cfg.pixel_format)
            self._set_node("offset_x", 0, clamp=True)
            self._set_node("offset_y", 0, clamp=True)
            if self.cfg.width > 0:
                self._set_node("width", self.cfg.width, clamp=True)
            if self.cfg.height > 0:
                self._set_node("height", self.cfg.height, clamp=True)
            if self.cfg.roi_offset_x:
                self._set_node("offset_x", self.cfg.roi_offset_x, clamp=True)
            if self.cfg.roi_offset_y:
                self._set_node("offset_y", self.cfg.roi_offset_y, clamp=True)

            logger.info("SpinCamera.open: [7] auto Off")
            self._set_node("exposure_auto", "Off")
            self._set_node("gain_auto", "Off")

            logger.info("SpinCamera.open: [8] gamma enable=%s val=%s", self.cfg.gamma_enable, self.cfg.gamma)
            self._set_node("gamma_enable", bool(self.cfg.gamma_enable))
            if self.cfg.gamma_enable:
                self._set_node("gamma", self.cfg.gamma, clamp=True)

            logger.info("SpinCamera.open: [9] acq_fps enable=%s val=%s",
                        self.cfg.acq_frame_rate_enable, self.cfg.acq_frame_rate)
            self._set_node("acq_fps_enable", bool(self.cfg.acq_frame_rate_enable))
            if self.cfg.acq_frame_rate_enable and self.cfg.acq_frame_rate > 0:
                self._set_node("acq_fps", self.cfg.acq_frame_rate, clamp=True)

            logger.info("SpinCamera.open: [10] exposure_ms=%s gain_db=%s", self.cfg.exposure_ms, self.cfg.gain_db)
            self._set_node("exposure_us", float(self.cfg.exposure_ms) * 1000.0, clamp=True)
            self._set_node("gain_db", self.cfg.gain_db, clamp=True)

            logger.info("SpinCamera.open: [11] blackfly extras clamp=%s blk=%s defect=%s",
                        self.cfg.black_level_clamping, self.cfg.black_level, self.cfg.defect_correction)
            self._set_node("black_clamp", bool(self.cfg.black_level_clamping))
            # Review N-2: 0.0 is a legitimate explicit black level; the
            # prior ``abs > 0`` guard skipped it and left the camera's
            # power-on default in place (typically non-zero on Blackfly).
            self._set_node("black_level", self.cfg.black_level, clamp=True)
            self._set_node("defect_corr", bool(self.cfg.defect_correction))

            logger.info("SpinCamera.open: [12] begin_acquisition()")
            self._cam.begin_acquisition()
            self._acquiring = True
            self._opened = True

            # Review B-3: retry stream_buffer if pre-begin set failed.
            # Post-begin the stream node is reliably available across all
            # Blackfly firmware we've tested.
            if not stream_buffer_set_ok:
                stream_buffer_set_ok = self._set_node(
                    "stream_buffer", self.cfg.buffer_handling
                )
                if stream_buffer_set_ok:
                    logger.info("SpinCamera.open: stream_buffer post-begin retry OK")
                else:
                    logger.warning(
                        "SpinCamera.open: stream_buffer unavailable both pre- and "
                        "post-begin_acquisition; AUDIT:parity-1 NOT applied -- live "
                        "view may lag."
                    )

            # Review N-1: report ACTUAL bound serial (not requested index).
            serial = self._get_node("device_serial", default="<unknown>")
            model = self._get_node("device_model", default="<unknown>")
            logger.info("SpinCamera: opened model=%s serial=%s", model, serial)
        except BaseException:
            logger.exception("SpinCamera.open: failed; tearing down")
            self._teardown_partial()
            raise

    def _teardown_partial(self) -> None:
        """Idempotent unwind invoked from ``open()`` and ``close()``. Each step
        is independently guarded so a partial-init failure (e.g. crash in
        begin_acquisition with a successful init_cam) still releases the C++
        objects cleanly. Logs at INFO -- this codepath is normal for the
        graceful-shutdown case."""
        if self._acquiring and self._cam is not None:
            try:
                self._cam.end_acquisition()
            except Exception as e:
                logger.info("SpinCamera: end_acquisition raised: %s", e)
        self._acquiring = False

        if self._init_cam_called and self._cam is not None:
            try:
                self._cam.deinit_cam()
            except Exception as e:
                logger.info("SpinCamera: deinit_cam raised: %s", e)
        self._init_cam_called = False

        if self._cam is not None:
            try:
                self._cam.release()
            except Exception as e:
                logger.info("SpinCamera: cam.release raised: %s", e)
        self._cam = None

        # SpinSystem has no .release() method in rotpy 0.2.1 (verified via
        # ROTPY_API.md). Dropping the reference triggers __dealloc__; the
        # known interpreter-shutdown noise ("sys.meta_path is None") is
        # cosmetic and out of scope here.
        self._system = None

    def close(self) -> None:
        """Idempotent across partial init (AUDIT:B2)."""
        self._teardown_partial()
        self._opened = False

    # --- frame acquisition ------------------------------------------------
    def grab(self) -> Optional[np.ndarray]:
        """Returns uint8 2-D ndarray (Mono8) per AUDIT:N2.

        ``None`` is the interruptibility-poll sentinel per AUDIT:B1, fired
        on three EXPECTED-transient conditions only:
            1. ``get_next_image`` raised ``SpinnakerAPIException`` with
               ``spin_error_code in _GRAB_TIMEOUT_CODES`` (i.e. -2007 timeout
               -- normal on every grab cycle when no frame is ready inside
               0.5s).
            2. ``img.get_status() != "no_error"`` -- image-level
               bandwidth/CRC/packet trouble.
            3. Buffer-size mismatch (defensive; should never happen on Mono8).

        Hard SDK errors (review TOP-1 fix): any non-timeout
        ``SpinnakerAPIException`` -- not_initialized / invalid_handle /
        access_denied / abort etc. -- RE-RAISES into the run loop, which
        emits an ``error`` signal and exits the thread. Previously grab()
        swallowed ALL SDK exceptions to None, which masked camera
        disconnects: the user saw the live view freeze with zero log
        output and no way to know the camera was gone.

        The frame is always an OWNED ndarray (``.base is None``) per AUDIT:S1.
        ``Image.get_image_data()`` returns a ``bytearray`` (verified
        2026-05-22) which is already a copy of the SDK buffer at the C++
        ``ImagePtr`` layer; the trailing ``.copy()`` after reshape removes
        the intermediate reference so the SDK buffer is fully released by
        ``img.release()``.
        """
        if self._cam is None or not self._acquiring:
            return None
        APIExc = self._SpinnakerAPIException
        try:
            img = self._cam.get_next_image(timeout=0.5)  # seconds per ROTPY_API.md
        except Exception as e:
            # TOP-1 fix: only the timeout code converts to None (the
            # interruptibility-poll path). Every other SDK exception
            # bubbles -- the run loop catches and decides transient vs
            # hard via _TRANSIENT_GRAB_ERROR_CODES.
            if APIExc is not None and isinstance(e, APIExc):
                code = int(getattr(e, "spin_error_code", 0) or 0)
                if code in _GRAB_TIMEOUT_CODES:
                    return None
            raise

        if img is None:
            return None
        try:
            status = img.get_status()
            if status != "no_error":
                # Image-level transient: bandwidth / CRC / packet trouble.
                # Caller throttles error spam; we just signal "no frame".
                return None

            h = img.get_height()
            w = img.get_width()
            data = img.get_image_data()  # bytearray, owned copy
            expected = h * w
            if len(data) < expected:
                logger.warning(
                    "SpinCamera.grab: buffer size mismatch %d < %d (pix_fmt=%s)",
                    len(data), expected, img.get_pix_fmt(),
                )
                return None
            # bytearray view -> reshape -> owned ndarray (.base is None).
            frame = np.frombuffer(data, dtype=np.uint8, count=expected).reshape(h, w).copy()
            # AUDIT:S1 dev assertion (review S-5): under -O this is a no-op;
            # under normal runs it enforces ownership on every emitted frame.
            assert frame.base is None, "AUDIT:S1 violation: frame still references intermediate"
            return frame
        finally:
            try:
                img.release()
            except Exception as e:
                logger.warning("SpinCamera.grab: img.release raised: %s", e)

    def reinit_aoi(self, width: int, height: int, start_x: int, start_y: int) -> None:
        """Change AOI on a (possibly) running camera. Stops acquisition,
        applies the four nodes (offsets zeroed first so a narrowing change
        doesn't briefly violate sensor bounds), restarts acquisition.

        ``_acquiring`` is set True ONLY after ``begin_acquisition`` actually
        returns (review B-2): otherwise a raise from ``begin_acquisition``
        would leave ``_acquiring=True`` while the stream is stopped, and the
        next ``grab()`` would block on ``get_next_image`` forever (0.5s
        timeout per call, but the camera stream is dead so every call
        times out)."""
        if self._cam is None:
            raise RuntimeError("reinit_aoi: camera not opened")
        was_acq = self._acquiring
        if was_acq:
            try:
                self._cam.end_acquisition()
            finally:
                self._acquiring = False
        self._set_node("offset_x", 0, clamp=True)
        self._set_node("offset_y", 0, clamp=True)
        if int(width) > 0:
            self._set_node("width", int(width), clamp=True)
        if int(height) > 0:
            self._set_node("height", int(height), clamp=True)
        self._set_node("offset_x", int(start_x), clamp=True)
        self._set_node("offset_y", int(start_y), clamp=True)
        if was_acq:
            # If this raises, _acquiring stays False (the truth). Caller
            # gets the exception and the run loop's _try wrapper logs and
            # continues; subsequent grabs return None until the camera is
            # restarted (close+open) or AOI re-applied successfully.
            self._cam.begin_acquisition()
            self._acquiring = True

    # --- direct setters used by the run loop (sub-step 2.5) ---------------
    # These mirror CameraThread.set_* slot names but operate on the live
    # camera handle. Run-loop pending-apply calls these from inside the
    # camera thread; never call them from the GUI thread.
    def set_exposure_ms(self, v: float) -> None:
        self._set_node("exposure_us", float(v) * 1000.0, clamp=True)

    def set_gain_db(self, v: float) -> None:
        self._set_node("gain_db", float(v), clamp=True)

    def set_gamma(self, v: float) -> None:
        self._set_node("gamma", float(v), clamp=True)

    def set_gamma_enable(self, v: bool) -> None:
        self._set_node("gamma_enable", bool(v))

    def set_pixel_format(self, v: str) -> None:
        """PixelFormat changes must happen while not acquiring; we stop &
        restart automatically (mirrors ``reinit_aoi``).

        ``_acquiring`` is set True ONLY after ``begin_acquisition`` returns
        (review B-2; same rationale as ``reinit_aoi``)."""
        if self._cam is None:
            return
        was_acq = self._acquiring
        if was_acq:
            try:
                self._cam.end_acquisition()
            finally:
                self._acquiring = False
        self._set_node("pixel_format", str(v))
        if was_acq:
            self._cam.begin_acquisition()
            self._acquiring = True

    def set_acquisition_frame_rate(self, v: float) -> None:
        self._set_node("acq_fps", float(v), clamp=True)

    def set_frame_rate_enable(self, v: bool) -> None:
        self._set_node("acq_fps_enable", bool(v))

    def set_packet_size(self, v: int) -> None:
        self._set_node("packet_size", int(v), clamp=True)

    def set_throughput_limit(self, v: int) -> None:
        self._set_node("throughput", int(v), clamp=True)

    def set_black_level(self, v: float) -> None:
        self._set_node("black_level", float(v), clamp=True)

    def set_black_level_clamping(self, v: bool) -> None:
        self._set_node("black_clamp", bool(v))

    def set_defect_correction(self, v: bool) -> None:
        self._set_node("defect_corr", bool(v))

    # --- introspection for the dock ---------------------------------------
    def get_camera_info(self) -> Dict[str, Any]:
        """Snapshot of every node the dock cares about, plus legacy
        uEye-shaped keys so an unmodified Step-1 dock degrades cleanly until
        Step 4. Tolerates missing nodes via ``_get_node`` / ``_get_range``.

        Review N-5: every coercion below funnels through ``_as_float`` /
        ``_as_int`` so a node returning an empty string (some Blackfly
        firmware does this for unset-but-implemented nodes) doesn't crash
        the dock with ``ValueError: could not convert string to float: ''``.
        """
        if self._cam is None:
            return {}

        def _as_float(v: Any, default: float = 0.0) -> float:
            try:
                if v is None or v == "":
                    return default
                return float(v)
            except (TypeError, ValueError):
                return default

        def _as_int(v: Any, default: int = 0) -> int:
            try:
                if v is None or v == "":
                    return default
                return int(v)
            except (TypeError, ValueError):
                return default

        sensor_w = self._get_node("sensor_width", default=0)
        sensor_h = self._get_node("sensor_height", default=0)
        roi_w = self._get_node("width", default=0)
        roi_h = self._get_node("height", default=0)
        roi_x = self._get_node("offset_x", default=0)
        roi_y = self._get_node("offset_y", default=0)

        roi_w_rng = self._get_range("width")
        roi_h_rng = self._get_range("height")
        roi_x_rng = self._get_range("offset_x")
        roi_y_rng = self._get_range("offset_y")

        exp_us = self._get_node("exposure_us", default=0.0)
        exp_rng = self._get_range("exposure_us")
        gain_db = self._get_node("gain_db", default=0.0)
        gain_rng = self._get_range("gain_db")
        gamma = self._get_node("gamma", default=1.0)
        gamma_rng = self._get_range("gamma")
        gamma_en = bool(self._get_node("gamma_enable", default=False))

        pix_fmt = self._get_node("pixel_format", default=self.cfg.pixel_format)
        pix_fmts = self._get_enum_entries("pixel_format")

        acq_fps = self._get_node("acq_fps", default=0.0)
        acq_fps_rng = self._get_range("acq_fps")
        acq_fps_en = bool(self._get_node("acq_fps_enable", default=False))

        tput = self._get_node("throughput", default=0)
        tput_rng = self._get_range("throughput")
        pkt = self._get_node("packet_size", default=self.cfg.gige_packet_size)
        pkt_d = self._get_node("packet_delay", default=self.cfg.gige_packet_delay)

        blk = self._get_node("black_level", default=0.0)
        blk_rng = self._get_range("black_level")
        blk_clamp = bool(self._get_node("black_clamp", default=True))
        defect = bool(self._get_node("defect_corr", default=False))

        serial = self._get_node("device_serial", default="<unknown>")
        model = self._get_node("device_model", default="<unknown>")

        exp_ms = float(exp_us) / 1000.0 if exp_us else 0.0
        exp_ms_min = (float(exp_rng[0]) / 1000.0) if exp_rng[0] is not None else 0.0
        exp_ms_max = (float(exp_rng[1]) / 1000.0) if exp_rng[1] is not None else 0.0
        exp_ms_inc = (float(exp_rng[2]) / 1000.0) if exp_rng[2] else 0.0

        info: Dict[str, Any] = {
            # --- Spinnaker-native --------------------------------------
            "sensor_width": int(sensor_w),
            "sensor_height": int(sensor_h),
            "roi_width": int(roi_w),
            "roi_height": int(roi_h),
            "roi_offset_x": int(roi_x),
            "roi_offset_y": int(roi_y),
            "roi_width_min": int(roi_w_rng[0]) if roi_w_rng[0] is not None else 0,
            "roi_width_max": int(roi_w_rng[1]) if roi_w_rng[1] is not None else int(sensor_w),
            "roi_width_inc": int(roi_w_rng[2]) if roi_w_rng[2] else 1,
            "roi_height_min": int(roi_h_rng[0]) if roi_h_rng[0] is not None else 0,
            "roi_height_max": int(roi_h_rng[1]) if roi_h_rng[1] is not None else int(sensor_h),
            "roi_height_inc": int(roi_h_rng[2]) if roi_h_rng[2] else 1,
            "roi_offset_x_min": int(roi_x_rng[0]) if roi_x_rng[0] is not None else 0,
            "roi_offset_x_max": int(roi_x_rng[1]) if roi_x_rng[1] is not None else int(sensor_w),
            "roi_offset_x_inc": int(roi_x_rng[2]) if roi_x_rng[2] else 1,
            "roi_offset_y_min": int(roi_y_rng[0]) if roi_y_rng[0] is not None else 0,
            "roi_offset_y_max": int(roi_y_rng[1]) if roi_y_rng[1] is not None else int(sensor_h),
            "roi_offset_y_inc": int(roi_y_rng[2]) if roi_y_rng[2] else 1,
            "exposure_us": float(exp_us),
            "exposure_us_min": float(exp_rng[0]) if exp_rng[0] is not None else 0.0,
            "exposure_us_max": float(exp_rng[1]) if exp_rng[1] is not None else 0.0,
            "exposure_us_inc": float(exp_rng[2]) if exp_rng[2] else 0.0,
            "gain_db": float(gain_db),
            "gain_db_min": float(gain_rng[0]) if gain_rng[0] is not None else 0.0,
            "gain_db_max": float(gain_rng[1]) if gain_rng[1] is not None else 0.0,
            "gamma": float(gamma),
            "gamma_min": float(gamma_rng[0]) if gamma_rng[0] is not None else 0.0,
            "gamma_max": float(gamma_rng[1]) if gamma_rng[1] is not None else 0.0,
            "gamma_enable": gamma_en,
            "pixel_format": str(pix_fmt),
            "pixel_formats": list(pix_fmts),
            "acq_fps": float(acq_fps),
            "acq_fps_min": float(acq_fps_rng[0]) if acq_fps_rng[0] is not None else 0.0,
            "acq_fps_max": float(acq_fps_rng[1]) if acq_fps_rng[1] is not None else 0.0,
            "acq_fps_enable": acq_fps_en,
            "throughput_limit": int(tput) if tput else 0,
            "throughput_limit_min": int(tput_rng[0]) if tput_rng[0] is not None else 0,
            "throughput_limit_max": int(tput_rng[1]) if tput_rng[1] is not None else 0,
            "packet_size": int(pkt),
            "packet_delay": int(pkt_d),
            "black_level": float(blk),
            "black_level_min": float(blk_rng[0]) if blk_rng[0] is not None else 0.0,
            "black_level_max": float(blk_rng[1]) if blk_rng[1] is not None else 0.0,
            "black_level_clamping": blk_clamp,
            "defect_correction": defect,
            "serial": str(serial),
            "model": str(model),

            # --- Legacy uEye-shaped keys (preserved until Step 4) -------
            "exposure": float(exp_ms),
            "exposure_min": float(exp_ms_min),
            "exposure_max": float(exp_ms_max),
            "exposure_inc": float(exp_ms_inc),
            # Legacy gain keys: int contract preserved (uEye was 0-100 int).
            # Step-4 dock redesign drops these in favor of float gain_db.
            "gain": int(round(gain_db)),
            "gain_min": int(round(gain_rng[0])) if gain_rng[0] is not None else 0,
            "gain_max": int(round(gain_rng[1])) if gain_rng[1] is not None else 0,
            "aoi_x": int(roi_x),
            "aoi_y": int(roi_y),
            "aoi_width": int(roi_w),
            "aoi_height": int(roi_h),
            "aoi_x_min": int(roi_x_rng[0]) if roi_x_rng[0] is not None else 0,
            "aoi_x_max": int(roi_x_rng[1]) if roi_x_rng[1] is not None else int(sensor_w),
            "aoi_y_min": int(roi_y_rng[0]) if roi_y_rng[0] is not None else 0,
            "aoi_y_max": int(roi_y_rng[1]) if roi_y_rng[1] is not None else int(sensor_h),
            "aoi_width_min": int(roi_w_rng[0]) if roi_w_rng[0] is not None else 0,
            "aoi_width_max": int(roi_w_rng[1]) if roi_w_rng[1] is not None else int(sensor_w),
            "aoi_width_inc": int(roi_w_rng[2]) if roi_w_rng[2] else 1,
            "aoi_height_min": int(roi_h_rng[0]) if roi_h_rng[0] is not None else 0,
            "aoi_height_max": int(roi_h_rng[1]) if roi_h_rng[1] is not None else int(sensor_h),
            "aoi_height_inc": int(roi_h_rng[2]) if roi_h_rng[2] else 1,
            "pixel_clock": 0,
            "pixel_clocks": [],
            "fps_min": float(acq_fps_rng[0]) if acq_fps_rng[0] is not None else 0.0,
            "fps_max": float(acq_fps_rng[1]) if acq_fps_rng[1] is not None else 0.0,
            "gain_boost": False,
        }
        return info


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
        """Set the loop-exit flag and request interruption.

        Interruption is bounded by ``SpinCamera.grab()``'s 0.5 s
        ``get_next_image`` timeout (AUDIT:B1) -- the timeout returns
        ``None`` and the loop re-checks ``isInterruptionRequested()`` at
        >= 2 Hz.

        WORST CASE (review S-3): if ``_apply_pending`` is mid-flight with a
        ``pixel_format`` or ``aoi`` change, the call to
        ``SpinCamera.set_pixel_format`` / ``reinit_aoi`` ``end_acquisition``
        + ``begin_acquisition`` synchronously; ``begin_acquisition`` can
        block on a GigE re-handshake for several seconds on a flaky link.
        ``stop()`` is honored only after that completes. Callers should
        size ``QThread.wait()`` accordingly (V9 fallback: log + leak rather
        than segfault if ``wait`` returns False)."""
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

    # --- error throttle (AUDIT:N1) ---------------------------------------
    def _err_throttled_emit(self, msg: str, *, throttle: bool = False) -> None:
        """Emit ``error`` signal with optional throttling.

        Non-throttled: emit immediately (every call hits the GUI).
        Throttled: first occurrence emits + arms a ``_ERROR_THROTTLE_WINDOW_S``
        window; further identical messages inside the window are counted but
        not emitted. Window expiry flushes a tail summary then re-arms."""
        if not throttle:
            self.error.emit(msg)
            return
        now = time.time()
        state = self._err_throttle_state.get(msg)
        if state is None:
            self.error.emit(msg)
            self._err_throttle_state[msg] = (0, now, now)
            return
        count, first_t, _last_t = state
        if (now - first_t) >= _ERROR_THROTTLE_WINDOW_S:
            if count > 0:
                self.error.emit(
                    f"{msg} (suppressed {count} more in last "
                    f"{_ERROR_THROTTLE_WINDOW_S:.0f}s)"
                )
            self.error.emit(msg)
            self._err_throttle_state[msg] = (0, now, now)
            return
        self._err_throttle_state[msg] = (count + 1, first_t, now)

    def _err_throttle_flush(self) -> None:
        """Flush any suppressed-count tail messages. Called on teardown."""
        for msg, (count, _f, _l) in list(self._err_throttle_state.items()):
            if count > 0:
                try:
                    self.error.emit(f"{msg} (suppressed {count} more total)")
                except Exception:
                    pass
        self._err_throttle_state.clear()

    # --- pending-apply pump (AUDIT:S2 apply-OUTSIDE-lock invariant) ------
    def _apply_pending(self, pending: Dict[str, Any]) -> bool:
        """Apply pending parameter changes to the live camera.

        Order is mandated by the plan (Step 2.5):
            pixel_format -> throughput/packet -> AOI ->
            acq_fps_enable + acq_fps -> gamma_enable + gamma ->
            exposure_ms -> gain_db -> decision-8 native ->
            ini_extras -> refresh_info

        Returns True iff at least one node-changing operation succeeded
        (caller re-emits ``camera_info_signal``). Each call is independently
        guarded -- one bad set doesn't kill the loop. Legacy uEye keys
        (pixel_clock, gain_boost, prioritize_exposure) are intentionally
        ignored here per Parity 2/4/5; their slot wrappers absorb the call
        and record a deprecation warning."""
        if self._cam is None:
            return False
        refresh = False

        def _try(name: str, fn) -> None:
            nonlocal refresh
            try:
                fn()
                refresh = True
            except Exception as e:
                logger.exception("CameraThread._apply_pending: %s failed", name)
                self._err_throttled_emit(f"{name}: {e}", throttle=False)

        if "pixel_format" in pending:
            _try("set_pixel_format",
                 lambda: self._cam.set_pixel_format(pending["pixel_format"]))

        if "device_link_throughput_limit" in pending:
            _try("set_throughput_limit",
                 lambda: self._cam.set_throughput_limit(pending["device_link_throughput_limit"]))
        if "gige_packet_size" in pending:
            _try("set_packet_size",
                 lambda: self._cam.set_packet_size(pending["gige_packet_size"]))

        if "aoi" in pending:
            w, h, sx, sy = pending["aoi"]
            _try("reinit_aoi",
                 lambda: self._cam.reinit_aoi(int(w), int(h), int(sx), int(sy)))

        # Review B-1: Spinnaker rejects AcquisitionFrameRate writes when
        # AcquisitionFrameRateEnable=False (node becomes unwritable). If a
        # single batch contains BOTH enable=False AND a rate change, the
        # rate write hits an unwritable node and is silently logged as
        # "rejected". Apply enable first, and SKIP the rate write when
        # enable is explicitly False in the same batch.
        new_enable_in_batch = pending.get("acq_frame_rate_enable", None)
        if new_enable_in_batch is not None:
            _try("set_frame_rate_enable",
                 lambda: self._cam.set_frame_rate_enable(new_enable_in_batch))
        if "acq_frame_rate" in pending and new_enable_in_batch is not False:
            _try("set_acquisition_frame_rate",
                 lambda: self._cam.set_acquisition_frame_rate(pending["acq_frame_rate"]))

        if "gamma_enable" in pending:
            _try("set_gamma_enable",
                 lambda: self._cam.set_gamma_enable(pending["gamma_enable"]))
        if "gamma" in pending:
            _try("set_gamma",
                 lambda: self._cam.set_gamma(pending["gamma"]))

        if "exposure_ms" in pending:
            _try("set_exposure_ms",
                 lambda: self._cam.set_exposure_ms(pending["exposure_ms"]))
        if "gain_db" in pending:
            _try("set_gain_db",
                 lambda: self._cam.set_gain_db(pending["gain_db"]))

        # Decision-8 Blackfly-native -- best-effort; missing nodes are
        # absorbed by SpinCamera._set_node and never raise here.
        if "black_level" in pending:
            _try("set_black_level",
                 lambda: self._cam.set_black_level(pending["black_level"]))
        if "black_level_clamping" in pending:
            _try("set_black_level_clamping",
                 lambda: self._cam.set_black_level_clamping(pending["black_level_clamping"]))
        if "defect_correction" in pending:
            _try("set_defect_correction",
                 lambda: self._cam.set_defect_correction(pending["defect_correction"]))

        if "ini_extras" in pending:
            _try("apply_ini_to_camera",
                 lambda: apply_ini_to_camera(self._cam, pending["ini_extras"]))

        if pending.get("refresh_info"):
            refresh = True

        # Legacy uEye keys absorbed but intentionally not applied
        # (pixel_clock / gain_boost / prioritize_exposure per Parity 2/4/5).
        return refresh

    # --- main loop --------------------------------------------------------
    def run(self) -> None:
        """Camera thread main loop.

        Lifecycle (AUDIT:B2 -- ``_cam`` construction + ``open`` BOTH inside
        the outer ``try:`` whose ``finally`` calls ``close``; no early
        return before finally):

        1. ``_cam = SpinCamera(self.cfg); _cam.open()``
        2. Emit initial ``camera_info_signal``
        3. Loop until ``self._running == False`` or ``isInterruptionRequested()``:
            a. Snapshot+clear ``_pending`` under ``_params_lock`` (AUDIT:S2)
            b. Apply pending changes via ``_apply_pending`` (OUTSIDE the lock)
            c. If anything node-changing happened: re-emit ``camera_info_signal``
            d. ``frame = _cam.grab()`` -- bounded by SpinCamera's 0.5s timeout
               per AUDIT:B1 (returns ``None`` on timeout / image-level
               transient; raises only for hard SDK errors)
            e. ``frame is None`` -> ``continue`` (interruptibility-poll path
               per AUDIT:B1; no emit, no sleep)
            f. ``new_frame.emit(frame)`` -> soft-throttle by ``cfg.max_fps``
        4. Hard exception path: throttled error emit. Transient SDK errors
           (in ``_TRANSIENT_GRAB_ERROR_CODES``) continue the loop; anything
           else re-raises into the outer ``finally``.
        5. ``finally`` -> idempotent ``_cam.close()``, flush error throttle,
           emit ``status('closed')``."""
        self._running = True
        self._cam = None
        APIExc = None  # captured after open() completes
        last_emit = 0.0
        try:
            self.status.emit("opening camera...")
            self._cam = SpinCamera(self.cfg)
            self._cam.open()
            APIExc = self._cam._SpinnakerAPIException
            self.status.emit("camera ready")

            try:
                self.camera_info_signal.emit(self._cam.get_camera_info())
            except Exception as e:
                logger.exception("CameraThread: initial get_camera_info failed")
                self._err_throttled_emit(f"get_camera_info: {e}", throttle=False)

            while self._running and not self.isInterruptionRequested():
                # 1. Snapshot+clear under lock; apply outside (AUDIT:S2)
                with QtCore.QMutexLocker(self._params_lock):
                    pending = dict(self._pending)
                    self._pending.clear()
                refresh_after = self._apply_pending(pending) if pending else False

                if refresh_after:
                    try:
                        self.camera_info_signal.emit(self._cam.get_camera_info())
                    except Exception as e:
                        logger.exception("CameraThread: refresh get_camera_info failed")
                        self._err_throttled_emit(f"get_camera_info: {e}", throttle=False)

                # 2. Grab a frame (bounded; interruptibility-poll via None)
                try:
                    frame = self._cam.grab()
                except Exception as e:
                    code = int(getattr(e, "spin_error_code", 0) or 0)
                    msg = getattr(e, "spin_msg", "") or str(e)
                    transient = (
                        APIExc is not None
                        and isinstance(e, APIExc)
                        and code in _TRANSIENT_GRAB_ERROR_CODES
                    )
                    self._err_throttled_emit(f"grab error: {msg}", throttle=transient)
                    if transient:
                        continue
                    # Hard SDK error -> escape to outer finally
                    raise

                if frame is None:
                    # Timeout or non-no_error image. AUDIT:B1 path:
                    # don't emit, don't sleep -- just re-poll the loop.
                    continue

                self.new_frame.emit(frame)

                # Soft FPS throttle (independent of camera FPS; protects GUI)
                now = time.time()
                interval = 1.0 / max(self.cfg.max_fps, 1e-6)
                slack_ms = int((interval - (now - last_emit)) * 1000.0)
                if slack_ms > 0:
                    self.msleep(slack_ms)
                last_emit = time.time()
        except Exception as e:
            # Review N-3: log the full traceback via the logger (not raw
            # stderr) so it lands in the lab's centralized log pipeline.
            logger.exception("CameraThread.run: terminal error in main loop")
            try:
                self._err_throttled_emit(f"run loop: {e}", throttle=False)
            except Exception:
                pass
        finally:
            if self._cam is not None:
                try:
                    self._cam.close()
                except Exception:
                    logger.exception("CameraThread: close raised in finally")
            try:
                self._err_throttle_flush()
            except Exception:
                pass
            try:
                self.status.emit("closed")
            except Exception:
                pass
            self._running = False


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
