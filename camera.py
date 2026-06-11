# camera.py
"""Spinnaker / ``rotpy`` camera driver for the rastering GUI.

Drop-in replacement for the legacy IDS uEye driver (preserved at
``camera_ueye.py``). Exposes the same Qt interface the GUI depends on --
``CameraThread`` (QThread) emitting ``new_frame`` / ``status`` / ``error`` /
``camera_info_signal``, the same parameter slots, and the same ``*_ini``
helpers -- but talks to a Teledyne FLIR Spinnaker camera through ``rotpy``.

================================  VERIFY ON HARDWARE  ========================
This module CANNOT be exercised off the lab machine: it needs the Spinnaker
SDK, ``rotpy`` (built per ``docs/ROTPY_BUILD.md``), and a connected camera.
Before trusting the GUI, run ``scripts/spin_smoke.py`` (the enumeration +
single-grab gate). GenICam node names (ExposureTime, Gain, Gamma,
AcquisitionFrameRate, Width/Height/OffsetX/OffsetY, DeviceLinkThroughputLimit,
DefectCorrectStaticEnable, ...) are the standard SFNC names used by Blackfly S
GigE models; every node access here is wrapped defensively so a missing/renamed
node degrades to a fallback rather than killing the acquisition thread.

uEye concepts that have no Spinnaker analog are mapped or stubbed (see
``CameraConfig`` and the relevant setters):
  * ``pixel_clock_mhz``  -> informational only; "pixel clock" has no Spinnaker
                            equivalent. The control is inert (logged, no-op).
  * ``enable_gain_boost``-> no Spinnaker analog; stored, no hardware effect.
  * ``use_freeze``       -> ignored; we always run continuous acquisition.
  * ``master_gain`` (0-100) is mapped linearly onto the camera's Gain (dB)
    range so the GUI's 0-100 slider keeps working unchanged.
=============================================================================
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore


# -----------------------------------------------------------------------------
# Spinnaker runtime bootstrap: make the SDK DLLs + GenTL producer discoverable
# before importing rotpy. Mirrors scripts/spin_smoke.py and the runbook (§2b).
# Safe no-op off Windows / when the SDK isn't at the expected path.
# -----------------------------------------------------------------------------
def _ensure_spinnaker_runtime() -> None:
    sdk_root = Path(r"C:\Program Files\Teledyne\Spinnaker")
    bin_dir = sdk_root / "bin64" / "vs2015"
    gentl_dir = sdk_root / "cti64" / "vs2015"
    if bin_dir.is_dir():
        try:
            os.add_dll_directory(str(bin_dir))  # py>=3.8 DLL search mechanism
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = f"{bin_dir};{os.environ.get('PATH', '')}"
    if gentl_dir.is_dir() and "GENICAM_GENTL64_PATH" not in os.environ:
        os.environ["GENICAM_GENTL64_PATH"] = str(gentl_dir)


_ensure_spinnaker_runtime()

try:
    from rotpy.system import SpinSystem
    from rotpy.camera import CameraList
except Exception as e:  # pragma: no cover - hardware/SDK dependent
    raise ImportError(
        "rotpy is required for camera.py -- build it from source per "
        "docs/ROTPY_BUILD.md into the rastering env."
    ) from e


@dataclass
class CameraConfig:
    """Initial camera configuration. Field names are kept identical to the
    legacy ``UEyeConfig`` so ``config.py`` / ``ui.py`` wiring is unchanged."""
    camera_id: int = 0
    width: int = 1280
    height: int = 1024
    pixel_clock_mhz: int = 10          # uEye-only; inert on Spinnaker
    exposure_ms: float = 30.0
    use_freeze: bool = True            # uEye-only; ignored (always continuous)
    emit_rgb: bool = False             # expand mono -> (H,W,3) for the UI
    max_fps: float = 15.0              # soft throttle on the emit loop
    roi_offset_x: int = 0
    roi_offset_y: int = 0
    master_gain: int = 10              # 0-100, mapped onto Gain[dB] range
    gamma: float = 1.6
    enable_gain_boost: bool = False    # uEye-only; no hardware effect
    target_fps: float = 20.0
    prioritize_exposure: bool = False


class _SpinCamera:
    """Thin ``rotpy`` wrapper: explicit open/close, single-frame grab, runtime
    parameter adjustment. Mirrors the legacy ``UEyeCamera`` surface so the
    thread/INI code reads the same."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.bitspixel = 8  # Mono8

        self._system: Optional[SpinSystem] = None
        self._cam_list = None
        self._cam = None  # rotpy Camera

        self._live_width = int(cfg.width)
        self._live_height = int(cfg.height)
        self._sensor_width = 0
        self._sensor_height = 0

        # Gain[dB] range, learned at open(); used to map the 0-100 UI slider.
        self._gain_min_db = 0.0
        self._gain_max_db = 1.0

        self._prioritize_exposure = cfg.prioritize_exposure
        self._cur_exposure_ms = float(cfg.exposure_ms)
        self._acquiring = False
        self._opened = False

    # ----- properties -------------------------------------------------------
    @property
    def live_width(self) -> int:
        return self._live_width

    @property
    def live_height(self) -> int:
        return self._live_height

    # ----- node access helpers (all defensive) ------------------------------
    # Every GenICam interaction funnels through these so an unexpected node
    # name/type fails soft (returns a fallback) instead of raising into the
    # acquisition loop. Setters return a bool so the caller can react.

    def _node(self, name: str):
        try:
            return getattr(self._cam.camera_nodes, name)
        except Exception:
            return None

    def _get_num(self, name: str, default=None):
        nd = self._node(name)
        if nd is None:
            return default
        try:
            return nd.get_node_value()
        except Exception:
            return default

    def _set_num(self, name: str, value) -> bool:
        nd = self._node(name)
        if nd is None:
            return False
        try:
            nd.set_node_value(value)
            return True
        except Exception:
            return False

    def _minmax(self, name: str) -> Tuple[Optional[float], Optional[float]]:
        nd = self._node(name)
        if nd is None:
            return None, None
        # rotpy float/int nodes expose get_min_value/get_max_value; tolerate
        # the shorter get_min/get_max spelling across versions.
        for lo, hi in (("get_min_value", "get_max_value"), ("get_min", "get_max")):
            try:
                return float(getattr(nd, lo)()), float(getattr(nd, hi)())
            except Exception:
                continue
        return None, None

    def _inc(self, name: str) -> Optional[int]:
        nd = self._node(name)
        if nd is None:
            return None
        for meth in ("get_inc_value", "get_inc"):
            try:
                return int(getattr(nd, meth)())
            except Exception:
                continue
        return None

    def _set_enum(self, name: str, str_value: str) -> bool:
        nd = self._node(name)
        if nd is None:
            return False
        try:
            nd.set_node_value_from_str(str_value)
            return True
        except Exception:
            return False

    # ----- lifecycle --------------------------------------------------------
    def open(self) -> None:
        self._system = SpinSystem()
        self._cam_list = CameraList.create_from_system(
            self._system, update_cams=True, update_interfaces=True)
        n = self._cam_list.get_size()
        if n == 0:
            raise RuntimeError("no Spinnaker cameras enumerated")
        idx = int(self.cfg.camera_id)
        if idx < 0 or idx >= n:
            raise RuntimeError(f"camera_id {idx} out of range (found {n})")
        self._cam = self._cam_list.create_camera_by_index(idx)
        self._cam.init_cam()

        # Continuous acquisition, Mono8 (the GUI consumes uint8 2-D frames).
        self._set_enum("AcquisitionMode", "Continuous")
        self._set_enum("PixelFormat", "Mono8")

        # Manual everything: no auto exposure / gain.
        self._set_enum("ExposureAuto", "Off")
        self._set_enum("ExposureMode", "Timed")
        self._set_enum("GainAuto", "Off")

        # Learn sensor extent + gain range.
        self._sensor_width = int(self._get_num("WidthMax", self.cfg.width) or self.cfg.width)
        self._sensor_height = int(self._get_num("HeightMax", self.cfg.height) or self.cfg.height)
        gmin, gmax = self._minmax("Gain")
        if gmin is not None and gmax is not None and gmax > gmin:
            self._gain_min_db, self._gain_max_db = gmin, gmax

        # Gamma on + initial value.
        self._set_num("GammaEnable", True)
        self.set_gamma(self.cfg.gamma)

        # AOI before acquisition (Width/Height/Offset are locked while running).
        self._apply_aoi(self.cfg.width, self.cfg.height,
                        self.cfg.roi_offset_x, self.cfg.roi_offset_y,
                        centered=True)

        # Frame rate + gain + exposure.
        self._apply_frame_rate_mode()
        self.set_master_gain(self.cfg.master_gain)
        self.set_exposure_ms(self.cfg.exposure_ms)

        self._cam.begin_acquisition()
        self._acquiring = True
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        try:
            if self._acquiring:
                self._cam.end_acquisition()
        except Exception:
            pass
        self._acquiring = False
        try:
            self._cam.deinit_cam()
        except Exception:
            pass
        # Drop refs child->parent so the SDK tears down cleanly.
        self._cam = None
        self._cam_list = None
        self._system = None
        self._opened = False

    # ----- AOI --------------------------------------------------------------
    def _align_down(self, value: int, inc: Optional[int]) -> int:
        if inc and inc > 1:
            return (int(value) // inc) * inc
        return int(value)

    def _apply_aoi(self, width: int, height: int,
                   offset_x: int = 0, offset_y: int = 0,
                   *, centered: bool) -> None:
        """Set Width/Height/OffsetX/OffsetY. ``centered`` mirrors the uEye
        center+offset convention; otherwise offset_x/y are absolute starts.
        Caller must ensure acquisition is stopped."""
        req_w = min(int(width), self._sensor_width or int(width))
        req_h = min(int(height), self._sensor_height or int(height))
        req_w = self._align_down(req_w, self._inc("Width"))
        req_h = self._align_down(req_h, self._inc("Height"))
        if req_w < 4 or req_h < 4:
            raise ValueError(f"AOI too small: {req_w}x{req_h}")

        # Zero the offsets first so a larger Width/Height is always permissible.
        self._set_num("OffsetX", 0)
        self._set_num("OffsetY", 0)
        if not self._set_num("Width", req_w):
            raise RuntimeError("failed to set Width")
        if not self._set_num("Height", req_h):
            raise RuntimeError("failed to set Height")

        if centered:
            off_x = (self._sensor_width - req_w) // 2 - int(offset_x)
            off_y = (self._sensor_height - req_h) // 2 - int(offset_y)
        else:
            off_x, off_y = int(offset_x), int(offset_y)
        off_x = max(0, min(off_x, max(0, self._sensor_width - req_w)))
        off_y = max(0, min(off_y, max(0, self._sensor_height - req_h)))
        off_x = self._align_down(off_x, self._inc("OffsetX"))
        off_y = self._align_down(off_y, self._inc("OffsetY"))
        self._set_num("OffsetX", off_x)
        self._set_num("OffsetY", off_y)

        self._live_width = int(self._get_num("Width", req_w) or req_w)
        self._live_height = int(self._get_num("Height", req_h) or req_h)

    def reinit_aoi(self, width: int, height: int, start_x: int, start_y: int) -> None:
        """Change the AOI at runtime. Width/Height/Offset are only writable
        while acquisition is stopped, so we bracket with end/begin."""
        was_acquiring = self._acquiring
        if was_acquiring:
            try:
                self._cam.end_acquisition()
            except Exception:
                pass
            self._acquiring = False
        self._apply_aoi(width, height, start_x, start_y, centered=False)
        if was_acquiring:
            self._cam.begin_acquisition()
            self._acquiring = True

    # ----- runtime setters --------------------------------------------------
    def set_exposure_ms(self, exposure_ms: float) -> None:
        us = float(exposure_ms) * 1000.0  # ExposureTime is microseconds
        lo, hi = self._minmax("ExposureTime")
        if lo is not None and hi is not None:
            us = max(lo, min(hi, us))
        if not self._set_num("ExposureTime", us):
            raise RuntimeError("failed to set ExposureTime")
        self._cur_exposure_ms = float(exposure_ms)

    def set_master_gain(self, gain: int) -> None:
        """Map the GUI's 0-100 slider linearly onto the camera Gain[dB] range."""
        g = max(0, min(100, int(gain)))
        db = self._gain_min_db + (g / 100.0) * (self._gain_max_db - self._gain_min_db)
        if not self._set_num("Gain", float(db)):
            raise RuntimeError("failed to set Gain")

    def _gain_to_slider(self, db: float) -> int:
        span = self._gain_max_db - self._gain_min_db
        if span <= 0:
            return 0
        return int(round((float(db) - self._gain_min_db) / span * 100.0))

    def set_gamma(self, gamma: float) -> None:
        g = float(gamma)
        lo, hi = self._minmax("Gamma")
        if lo is not None and hi is not None:
            g = max(lo, min(hi, g))
        if not self._set_num("Gamma", g):
            raise RuntimeError("failed to set Gamma")

    def set_frame_rate(self, fps: float) -> None:
        # Frame-rate control must be enabled before AcquisitionFrameRate is
        # writable; some models also expose AcquisitionFrameRateAuto.
        self._set_enum("AcquisitionFrameRateAuto", "Off")
        if not (self._set_num("AcquisitionFrameRateEnable", True)
                or self._set_num("AcquisitionFrameRateEnabled", True)):
            pass  # node absent on some models; best-effort
        f = float(fps)
        lo, hi = self._minmax("AcquisitionFrameRate")
        if lo is not None and hi is not None:
            f = max(lo, min(hi, f))
        if not self._set_num("AcquisitionFrameRate", f):
            raise RuntimeError("failed to set AcquisitionFrameRate")

    def _apply_frame_rate_mode(self) -> None:
        """exposure-priority -> let the camera free-run at max exposure headroom
        (disable the frame-rate cap); fps-priority -> cap at target_fps."""
        try:
            if self._prioritize_exposure:
                if not self._set_num("AcquisitionFrameRateEnable", False):
                    self._set_num("AcquisitionFrameRateEnabled", False)
            else:
                self.set_frame_rate(self.cfg.target_fps)
        except Exception:
            pass

    def set_pixel_clock(self, mhz: int) -> None:
        # No Spinnaker analog. Kept so the GUI slot exists; intentionally inert.
        # (uEye "pixel clock" traded bandwidth for FPS/exposure range; on GigE
        # that lever is DeviceLinkThroughputLimit, left at its default here.)
        return

    def set_gain_boost(self, enabled: bool) -> None:
        # No Spinnaker analog on Blackfly S; stored only, no hardware effect.
        return

    # ----- info -------------------------------------------------------------
    def get_camera_info(self) -> Dict[str, Any]:
        """Query current parameters + valid ranges. Keys match the legacy uEye
        dict exactly, because ``camera_settings_dock`` was written against it."""
        info: Dict[str, Any] = {}
        info["sensor_width"] = self._sensor_width
        info["sensor_height"] = self._sensor_height

        # Pixel clock: not applicable -> single inert entry so the combo is sane.
        info["pixel_clock"] = int(self.cfg.pixel_clock_mhz)
        info["pixel_clocks"] = [int(self.cfg.pixel_clock_mhz)]

        # Exposure (ms).
        emin, emax = self._minmax("ExposureTime")
        ecur = self._get_num("ExposureTime", None)
        einc = self._inc("ExposureTime")
        info["exposure_min"] = (emin / 1000.0) if emin is not None else 0.01
        info["exposure_max"] = (emax / 1000.0) if emax is not None else 1000.0
        info["exposure_inc"] = (float(einc) / 1000.0) if einc else 0.01
        info["exposure"] = (float(ecur) / 1000.0) if ecur is not None else self._cur_exposure_ms

        # FPS.
        fmin, fmax = self._minmax("AcquisitionFrameRate")
        info["fps_min"] = round(fmin, 2) if fmin is not None else 0.1
        info["fps_max"] = round(fmax, 2) if fmax is not None else 100.0
        fcur = self._get_num("AcquisitionFrameRate", None)
        info["fps"] = round(float(fcur), 2) if fcur is not None else 0.0

        # Gain reported on the 0-100 UI scale.
        info["gain_min"] = 0
        info["gain_max"] = 100
        gcur_db = self._get_num("Gain", None)
        info["gain"] = self._gain_to_slider(gcur_db) if gcur_db is not None else 0

        info["gain_boost"] = False  # no Spinnaker analog

        # Gamma.
        gmin, gmax = self._minmax("Gamma")
        gcur = self._get_num("Gamma", None)
        info["gamma"] = float(gcur) if gcur is not None else 1.0
        info["gamma_min"] = gmin if gmin is not None else 0.25
        info["gamma_max"] = gmax if gmax is not None else 4.0

        # AOI (read back).
        info["aoi_x"] = int(self._get_num("OffsetX", 0) or 0)
        info["aoi_y"] = int(self._get_num("OffsetY", 0) or 0)
        info["aoi_width"] = int(self._get_num("Width", self._live_width) or self._live_width)
        info["aoi_height"] = int(self._get_num("Height", self._live_height) or self._live_height)
        return info

    # ----- grab -------------------------------------------------------------
    def grab(self) -> np.ndarray:
        if not self._opened:
            raise RuntimeError("Camera not opened")
        timeout_s = max(5.0, self._cur_exposure_ms / 1000.0 * 2.0 + 1.0)
        img = self._cam.get_next_image(timeout=timeout_s)
        try:
            if img.is_incomplete():
                raise RuntimeError("image incomplete")
            h = img.get_height()
            w = img.get_width()
            data = img.get_image_data()  # view into the SDK buffer
            flat = np.frombuffer(bytes(data), dtype=np.uint8)
            if flat.size != h * w:
                raise RuntimeError(
                    f"buffer size mismatch: {flat.size} != {h*w} (PixelFormat not Mono8?)")
            # Copy off the SDK buffer BEFORE releasing it back to the pool.
            frame = np.ascontiguousarray(flat.reshape(h, w))
        finally:
            try:
                img.release()
            except Exception:
                pass
        if self.cfg.emit_rgb:
            frame = np.dstack((frame, frame, frame))
        return frame


# =====================================================================
# Camera thread with unified pending-parameter pattern
# =====================================================================

class CameraThread(QtCore.QThread):
    new_frame = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    # Emitted after open and after any parameter change that affects ranges.
    camera_info_signal = QtCore.pyqtSignal(dict)

    # Transient grab conditions (timeout / incomplete) get throttled so a
    # momentary GigE hiccup doesn't flood the GUI log.
    _TRANSIENT_GRAB_MARKERS = ("incomplete", "timed out", "timeout")
    _ERROR_THROTTLE_WINDOW_S = 5.0

    def __init__(self, cfg: Optional[CameraConfig] = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg or CameraConfig()
        self._cam: Optional[_SpinCamera] = None
        self._running = False

        self._params_lock = QtCore.QMutex()
        self._pending: Dict[str, Any] = {}

        # msg_key -> (count, first_ts, last_ts) for error de-duplication.
        self._err_throttle_state: Dict[str, Tuple[int, float, float]] = {}

    # ----- thread-safe parameter slots (called from the UI thread) ----------
    @QtCore.pyqtSlot(float)
    def set_exposure_ms(self, v: float) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["exposure"] = float(v)

    @QtCore.pyqtSlot(int)
    def set_master_gain(self, v: int) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["gain"] = int(v)

    @QtCore.pyqtSlot(float)
    def set_gamma(self, v: float) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["gamma"] = float(v)

    @QtCore.pyqtSlot(int)
    def set_pixel_clock(self, v: int) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["pixel_clock"] = int(v)

    @QtCore.pyqtSlot(bool)
    def set_gain_boost(self, v: bool) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["gain_boost"] = bool(v)

    def request_aoi_change(self, width: int, height: int, start_x: int, start_y: int) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["aoi"] = (int(width), int(height), int(start_x), int(start_y))

    @QtCore.pyqtSlot(float)
    def set_target_fps(self, v: float) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["fps"] = float(v)

    @QtCore.pyqtSlot(bool)
    def set_prioritize_exposure(self, v: bool) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["prioritize_exposure"] = bool(v)

    def request_ini_extras(self, ini_path: str) -> None:
        """Apply .ini extras (hotpixel, hw gamma, exact AOI) on the cam thread."""
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["ini_extras"] = str(ini_path)

    def request_info_refresh(self) -> None:
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["refresh_info"] = True

    def stop(self) -> None:
        self._running = False
        self.requestInterruption()

    # ----- error throttling (camera-thread only) ----------------------------
    def _err_throttled_emit(self, msg: str, *, throttle: bool) -> None:
        if not throttle:
            self.error.emit(msg)
            return
        now = time.time()
        state = self._err_throttle_state.get(msg)
        if state is None:
            self._err_throttle_state[msg] = (1, now, now)
            self.error.emit(msg)
        else:
            count, first, _last = state
            self._err_throttle_state[msg] = (count + 1, first, now)

    def _err_throttle_flush(self) -> None:
        now = time.time()
        expired = []
        for key, (count, first, last) in self._err_throttle_state.items():
            if (now - last) >= self._ERROR_THROTTLE_WINDOW_S and count > 1:
                duration = max(0.001, last - first)
                self.error.emit(f"{key} (x{count} over {duration:.1f}s)")
                expired.append(key)
        for key in expired:
            del self._err_throttle_state[key]

    # ----- run loop ---------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self._cam = _SpinCamera(self.cfg)
        try:
            self._cam.open()
            self.status.emit("Spinnaker camera opened.")
        except Exception as e:
            self.error.emit(f"Camera open failed: {e}")
            return

        try:
            self.camera_info_signal.emit(self._cam.get_camera_info())
        except Exception:
            pass

        dt_min = 1.0 / max(float(self.cfg.max_fps), 0.1)
        t_last = 0.0

        try:
            while self._running and not self.isInterruptionRequested():
                # ---- apply pending parameter changes ----
                with QtCore.QMutexLocker(self._params_lock):
                    pending = dict(self._pending)
                    self._pending.clear()

                need_info_update = False

                if "prioritize_exposure" in pending:
                    self._cam._prioritize_exposure = pending["prioritize_exposure"]
                    try:
                        self._cam._apply_frame_rate_mode()
                        need_info_update = True
                    except Exception as e:
                        self.error.emit(f"Timing mode set failed: {e}")

                if "pixel_clock" in pending:
                    # Inert on Spinnaker; acknowledge so the UI isn't silent.
                    self._cam.set_pixel_clock(pending["pixel_clock"])
                    self.status.emit("Pixel clock is uEye-only; ignored on Spinnaker.")

                if "fps" in pending:
                    try:
                        self._cam.set_frame_rate(pending["fps"])
                        self._cam.cfg = _dc_replace(self._cam.cfg, target_fps=pending["fps"])
                        need_info_update = True
                    except Exception as e:
                        self.error.emit(f"FPS set failed: {e}")

                if "aoi" in pending:
                    w, h, sx, sy = pending["aoi"]
                    try:
                        self._cam.reinit_aoi(w, h, sx, sy)
                        need_info_update = True
                        self.status.emit(
                            f"AOI: {self._cam.live_width}x{self._cam.live_height} at ({sx},{sy})")
                    except Exception as e:
                        self.error.emit(f"AOI change failed: {e}")

                if "gain" in pending:
                    try:
                        self._cam.set_master_gain(pending["gain"])
                    except Exception as e:
                        self.error.emit(f"Gain set failed: {e}")

                if "gain_boost" in pending:
                    self._cam.set_gain_boost(pending["gain_boost"])

                if "gamma" in pending:
                    try:
                        self._cam.set_gamma(pending["gamma"])
                    except Exception as e:
                        self.error.emit(f"Gamma set failed: {e}")

                if "exposure" in pending:
                    try:
                        exp_val = pending["exposure"]
                        if self._cam._prioritize_exposure and exp_val > 0:
                            # Drop the frame-rate cap enough to fit the exposure.
                            try:
                                self._cam.set_frame_rate(1000.0 / exp_val)
                            except Exception:
                                pass
                        self._cam.set_exposure_ms(exp_val)
                        need_info_update = True
                    except Exception as e:
                        self.error.emit(f"Exposure set failed: {e}")

                if "ini_extras" in pending:
                    try:
                        apply_ini_to_camera(self._cam, pending["ini_extras"])
                        need_info_update = True
                    except Exception as e:
                        self.error.emit(f"INI extras apply failed: {e}")

                if "refresh_info" in pending:
                    need_info_update = True

                if need_info_update:
                    try:
                        self.camera_info_signal.emit(self._cam.get_camera_info())
                    except Exception:
                        pass

                # ---- grab ----
                try:
                    frame = self._cam.grab()
                except Exception as e:
                    msg = f"Frame grab failed: {e}"
                    low = str(e).lower()
                    transient = any(m in low for m in self._TRANSIENT_GRAB_MARKERS)
                    self._err_throttled_emit(msg, throttle=transient)
                    self._err_throttle_flush()
                    continue

                self.new_frame.emit(frame)
                self._err_throttle_flush()

                # ---- soft throttle ----
                now = time.time()
                if t_last != 0.0:
                    dt = now - t_last
                    if dt < dt_min:
                        time.sleep(dt_min - dt)
                t_last = time.time()

        finally:
            try:
                self._cam.close()
            except Exception:
                pass
            self.status.emit("Spinnaker camera closed.")


# =====================================================================
# .ini config (kept Cockpit-compatible so old/new files interoperate)
# =====================================================================

def load_camera_config_from_ini(ini_path: str, **overrides) -> CameraConfig:
    """Parse a (uEye Cockpit-style) .ini and return a CameraConfig. Fields not
    present fall back to defaults; ``overrides`` take final precedence. The
    format is preserved so existing camera_params.ini files keep loading."""
    import configparser

    cp = configparser.ConfigParser()
    cp.read(ini_path, encoding="utf-8-sig")

    def _get(section: str, key: str, fallback=None):
        try:
            return cp.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def _getint(section: str, key: str, fallback: int = 0) -> int:
        v = _get(section, key)
        return int(v) if v is not None else fallback

    def _getfloat(section: str, key: str, fallback: float = 0.0) -> float:
        v = _get(section, key)
        return float(v) if v is not None else fallback

    kwargs: dict = {}
    kwargs["width"] = _getint("Image size", "Width", 1280)
    kwargs["height"] = _getint("Image size", "Height", 1024)
    kwargs["pixel_clock_mhz"] = _getint("Timing", "Pixelclock", 10)

    fps = _getfloat("Timing", "Framerate", 0)
    if fps > 0:
        kwargs["target_fps"] = fps
        kwargs["max_fps"] = fps + 5.0

    target_fps = _getfloat("Timing", "TargetFPS", 0)
    if target_fps > 0:
        kwargs["target_fps"] = target_fps
        kwargs["max_fps"] = target_fps + 5.0
    timing_mode = _get("Timing", "TimingMode", "fps")
    kwargs["prioritize_exposure"] = (timing_mode == "exposure")

    kwargs["exposure_ms"] = _getfloat("Timing", "Exposure", 30.0)
    kwargs["master_gain"] = _getint("Gain", "Master", 10)
    kwargs["enable_gain_boost"] = bool(_getint("Gain", "GainBoost", 0))

    gamma = _getfloat("Parameters", "Gamma", 1.0)
    if gamma > 0:
        kwargs["gamma"] = gamma

    kwargs["roi_offset_x"] = _getint("Image size", "Start X", 0)
    kwargs["roi_offset_y"] = _getint("Image size", "Start Y", 0)

    kwargs.update(overrides)
    return CameraConfig(**kwargs)


def apply_ini_to_camera(cam: "_SpinCamera", ini_path: str) -> None:
    """Apply .ini settings beyond CameraConfig fields -- hotpixel correction,
    hardware gamma, and the .ini's exact AOI. Call AFTER cam.open(). Maps the
    uEye .ini keys onto Spinnaker nodes; every step is best-effort."""
    import configparser

    cp = configparser.ConfigParser()
    cp.read(ini_path, encoding="utf-8-sig")

    def _getint(section, key, fallback=0):
        try:
            return int(cp.get(section, key))
        except Exception:
            return fallback

    # Hotpixel -> static defect correction.
    hp_mode = _getint("Parameters", "Hotpixel Mode", -1)
    if hp_mode >= 0:
        cam._set_num("DefectCorrectStaticEnable", bool(hp_mode))

    # Hardware gamma -> GammaEnable.
    if _getint("Parameters", "Hardware Gamma", 0):
        cam._set_num("GammaEnable", True)

    # Exact AOI from the .ini.
    ini_sx = _getint("Image size", "Start X", -1)
    ini_sy = _getint("Image size", "Start Y", -1)
    ini_w = _getint("Image size", "Width", -1)
    ini_h = _getint("Image size", "Height", -1)
    if ini_sx >= 0 and ini_sy >= 0 and ini_w > 0 and ini_h > 0:
        cam.reinit_aoi(ini_w, ini_h, ini_sx, ini_sy)


def save_settings_to_ini(ini_path: str, settings: dict) -> None:
    """Write camera + display settings to a Cockpit-compatible .ini. ``settings``
    is the dict from ``CameraSettingsDock.get_current_settings()``. Display-only
    fields live in an extra [Display] section our loader reads back."""
    import configparser

    cp = configparser.ConfigParser()
    try:
        cp.read(ini_path, encoding="utf-8-sig")
    except Exception:
        pass

    for section in ("Image size", "Timing", "Gain", "Parameters", "Display"):
        if not cp.has_section(section):
            cp.add_section(section)

    cp.set("Image size", "Width", str(settings.get("aoi_width", 1280)))
    cp.set("Image size", "Height", str(settings.get("aoi_height", 1024)))
    cp.set("Image size", "Start X", str(settings.get("aoi_x", 0)))
    cp.set("Image size", "Start Y", str(settings.get("aoi_y", 0)))

    cp.set("Timing", "Pixelclock", str(settings.get("pixel_clock", 10)))
    exposure = settings.get("exposure", 30.0)
    cp.set("Timing", "Exposure", f"{exposure:.6f}")
    if exposure > 0:
        cp.set("Timing", "Framerate", f"{1000.0 / exposure:.6f}")
    cp.set("Timing", "TimingMode", settings.get("timing_mode", "fps"))
    cp.set("Timing", "TargetFPS", f"{settings.get('target_fps', 20.0):.6f}")

    cp.set("Gain", "Master", str(settings.get("gain", 0)))
    cp.set("Gain", "GainBoost", str(int(settings.get("gain_boost", False))))

    cp.set("Parameters", "Gamma", f"{settings.get('gamma', 1.0):.6f}")

    cp.set("Display", "rotation_k", str(settings.get("rotation_k", 0)))
    cp.set("Display", "flip_x", str(int(settings.get("flip_x", False))))
    cp.set("Display", "flip_y", str(int(settings.get("flip_y", False))))

    with open(ini_path, "w", encoding="utf-8") as f:
        cp.write(f)


def _load_display_settings_from_ini(ini_path: str) -> dict:
    """Read [Display] (rotation_k, flip_x, flip_y) from a saved .ini."""
    import configparser
    cp = configparser.ConfigParser()
    try:
        cp.read(ini_path, encoding="utf-8-sig")
    except Exception:
        return {}

    out: dict = {}
    try:
        out["rotation_k"] = int(cp.get("Display", "rotation_k"))
    except Exception:
        pass
    try:
        out["flip_x"] = bool(int(cp.get("Display", "flip_x")))
    except Exception:
        pass
    try:
        out["flip_y"] = bool(int(cp.get("Display", "flip_y")))
    except Exception:
        pass
    return out


# Standalone enumeration/preview sanity check (no GUI).
def _standalone() -> None:
    cfg = CameraConfig()
    cam = _SpinCamera(cfg)
    cam.open()
    try:
        print("Camera info:", cam.get_camera_info())
        for _ in range(10):
            frame = cam.grab()
            print("frame:", frame.dtype, frame.shape)
    finally:
        cam.close()


if __name__ == "__main__":
    _standalone()
