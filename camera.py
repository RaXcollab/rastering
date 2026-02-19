# camera.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

try:
    from pyueye import ueye
except Exception as e:
    raise ImportError("pyueye is required for camera.py") from e


@dataclass
class UEyeConfig:
    camera_id: int = 0
    width: int = 1280
    height: int = 1024
    pixel_clock_mhz: int = 10          # Controls fps and bandwidth
    exposure_ms: float = 30.0
    use_freeze: bool = True            # FreezeVideo loop avoids stale buffers
    emit_rgb: bool = False             # UI expects RGB-ish; uEye often gives BGR
    max_fps: float = 15.0              # soft throttle
    roi_offset_x: int = 0
    roi_offset_y: int = 0
    master_gain: int = 10           # 0-100
    gamma: float = 1.6              # Will be converted to 160
    enable_gain_boost: bool = False
    target_fps: float = 20.0


class UEyeCamera:
    """
    Thin uEye wrapper with explicit init/close, single-frame grab,
    and runtime parameter adjustment.
    """
    def __init__(self, cfg: UEyeConfig):
        self.cfg = cfg
        self.hcam = ueye.HIDS(int(cfg.camera_id))

        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()
        
        # Use 8-bit Mono to save bandwidth (3x faster than BGR8)
        self.bitspixel = 8  
        
        # Live dimensions (updated on AOI change; cfg is the *initial* config)
        self._live_width = int(cfg.width)
        self._live_height = int(cfg.height)
        self._lineinc = self._live_width * int((self.bitspixel + 7) / 8)

        # Sensor max dimensions (populated in open())
        self._sensor_width = 0
        self._sensor_height = 0

        self._opened = False

    @property
    def live_width(self) -> int:
        return self._live_width

    @property
    def live_height(self) -> int:
        return self._live_height

    def open(self) -> None:
        ret = ueye.is_InitCamera(self.hcam, None)
        if ret != 0:
            raise RuntimeError(f"is_InitCamera failed: {ret}")

        # ------------------------------------------------------------------
        # 0. Read sensor info
        # ------------------------------------------------------------------
        sinfo = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(self.hcam, sinfo)
        self._sensor_width = int(sinfo.nMaxWidth)
        self._sensor_height = int(sinfo.nMaxHeight)

        # ------------------------------------------------------------------
        # 1. Color Mode
        # ------------------------------------------------------------------
        ueye.is_SetColorMode(self.hcam, ueye.IS_CM_MONO8)

        # ------------------------------------------------------------------
        # 2. Gain, Boost & Gamma 
        # ------------------------------------------------------------------
        zero_val = ueye.double(0)
        ueye.is_SetAutoParameter(self.hcam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, zero_val, zero_val)
        ueye.is_SetAutoParameter(self.hcam, ueye.IS_SET_ENABLE_AUTO_GAIN, zero_val, zero_val)

        ueye.is_SetHardwareGain(self.hcam, 
                                ueye.int(self.cfg.master_gain), 
                                ueye.IS_IGNORE_PARAMETER, 
                                ueye.IS_IGNORE_PARAMETER, 
                                ueye.IS_IGNORE_PARAMETER)

        gamma_int = ueye.int(int(self.cfg.gamma * 100))
        ueye.is_Gamma(self.hcam, ueye.IS_GAMMA_CMD_SET, gamma_int, ueye.sizeof(gamma_int))
        
        n_mode = ueye.int(ueye.IS_AUTO_BLACKLEVEL_ON)
        ueye.is_Blacklevel(self.hcam, ueye.IS_BLACKLEVEL_CMD_SET_MODE, n_mode, 0)

        if self.cfg.enable_gain_boost:
            try:
                ueye.is_SetGainBoost(self.hcam, ueye.IS_SET_GAINBOOST_ON)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 3. Timing: Clock -> FPS -> Exposure (Strict Order)
        # ------------------------------------------------------------------
        pclk = ueye.int(int(self.cfg.pixel_clock_mhz))
        ueye.is_PixelClock(self.hcam, ueye.IS_PIXELCLOCK_CMD_SET, pclk, ueye.sizeof(pclk))

        new_fps = ueye.double(self.cfg.target_fps)
        actual_fps = ueye.double(0)
        ueye.is_SetFrameRate(self.hcam, new_fps, actual_fps)

        # ------------------------------------------------------------------
        # 4. AOI setup
        # ------------------------------------------------------------------
        self._setup_aoi(self.cfg.width, self.cfg.height,
                        self.cfg.roi_offset_x, self.cfg.roi_offset_y,
                        use_offsets=True)

        # ------------------------------------------------------------------
        # 5. Image memory
        # ------------------------------------------------------------------
        self._alloc_memory()

        # Start continuous capture if not using freeze mode
        if not self.cfg.use_freeze:
            ret = ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)
            if ret != 0:
                raise RuntimeError(f"is_CaptureVideo failed: {ret}")

        # ------------------------------------------------------------------
        # 6. Exposure (after timing + AOI are configured)
        # ------------------------------------------------------------------
        self.set_exposure_ms(self.cfg.exposure_ms)

        self._opened = True

    # ------------------------------------------------------------------
    # AOI + memory helpers
    # ------------------------------------------------------------------

    def _setup_aoi(self, width: int, height: int,
                   offset_x: int = 0, offset_y: int = 0,
                   *, use_offsets: bool = True,
                   start_x: Optional[int] = None,
                   start_y: Optional[int] = None) -> None:
        """
        Set the AOI on the sensor.

        Two modes:
          use_offsets=True:  center the AOI and apply offset_x/y from center
          use_offsets=False: use absolute start_x/start_y directly
        """
        req_w = int(width)
        req_h = int(height)

        # Clamp dimensions to sensor
        req_w = min(req_w, self._sensor_width)
        req_h = min(req_h, self._sensor_height)
        # Align to 4 pixels
        req_w = (req_w // 4) * 4
        req_h = (req_h // 4) * 4
        if req_w < 4 or req_h < 4:
            raise ValueError(f"AOI too small: {req_w}x{req_h}")

        if use_offsets and start_x is None:
            # Centered + offset approach (legacy)
            off_x = (self._sensor_width - req_w) // 2 - int(offset_x)
            off_y = (self._sensor_height - req_h) // 2 - int(offset_y)
        else:
            off_x = int(start_x or 0)
            off_y = int(start_y or 0)

        # Clamp to sensor bounds
        off_x = max(0, min(off_x, self._sensor_width - req_w))
        off_y = max(0, min(off_y, self._sensor_height - req_h))
        # Align to 4 pixels
        off_x = (off_x // 4) * 4
        off_y = (off_y // 4) * 4

        rect = ueye.IS_RECT()
        rect.s32X = ueye.int(off_x)
        rect.s32Y = ueye.int(off_y)
        rect.s32Width = ueye.int(req_w)
        rect.s32Height = ueye.int(req_h)
        ret = ueye.is_AOI(self.hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect))
        if ret != 0:
            raise RuntimeError(f"is_AOI set failed: {ret}")

        self._live_width = req_w
        self._live_height = req_h
        self._lineinc = req_w * int((self.bitspixel + 7) / 8)

    def _alloc_memory(self) -> None:
        """Allocate + activate image memory for current AOI dimensions."""
        ret = ueye.is_AllocImageMem(
            self.hcam,
            ueye.int(self._live_width),
            ueye.int(self._live_height),
            ueye.int(self.bitspixel),
            self.mem_ptr,
            self.mem_id
        )
        if ret != 0:
            raise RuntimeError(f"is_AllocImageMem failed: {ret}")

        ret = ueye.is_SetImageMem(self.hcam, self.mem_ptr, self.mem_id)
        if ret != 0:
            raise RuntimeError(f"is_SetImageMem failed: {ret}")

    def _free_memory(self) -> None:
        """Free current image memory (best-effort)."""
        try:
            ueye.is_FreeImageMem(self.hcam, self.mem_ptr, self.mem_id)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Runtime parameter setters
    # ------------------------------------------------------------------

    def set_exposure_ms(self, exposure_ms: float) -> None:
        exp = ueye.double(float(exposure_ms))
        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp, ueye.sizeof(exp))
        if ret != 0:
            raise RuntimeError(f"is_Exposure set failed: {ret}")

    def set_master_gain(self, gain: int) -> None:
        gain = max(0, min(100, int(gain)))
        ueye.is_SetHardwareGain(self.hcam,
                                ueye.int(gain),
                                ueye.IS_IGNORE_PARAMETER,
                                ueye.IS_IGNORE_PARAMETER,
                                ueye.IS_IGNORE_PARAMETER)

    def set_gamma(self, gamma: float) -> None:
        gamma_int = ueye.int(max(1, min(1000, int(float(gamma) * 100))))
        ueye.is_Gamma(self.hcam, ueye.IS_GAMMA_CMD_SET, gamma_int, ueye.sizeof(gamma_int))

    def set_pixel_clock(self, mhz: int) -> None:
        """
        Set pixel clock. This changes the available FPS and exposure ranges.
        After setting, re-sets FPS to minimum (longest exposure headroom).
        """
        pclk = ueye.int(int(mhz))
        ret = ueye.is_PixelClock(self.hcam, ueye.IS_PIXELCLOCK_CMD_SET, pclk, ueye.sizeof(pclk))
        if ret != 0:
            raise RuntimeError(f"is_PixelClock set failed: {ret}")

        # After changing pixel clock, reset frame rate for max exposure headroom
        try:
            min_time = ueye.double()
            max_time = ueye.double()
            inc_time = ueye.double()
            ueye.is_GetFrameTimeRange(self.hcam, min_time, max_time, inc_time)
            safe_fps = 1.0 / float(max_time) if float(max_time) > 0 else 1.0
            new_fps = ueye.double(safe_fps)
            actual_fps = ueye.double(0)
            ueye.is_SetFrameRate(self.hcam, new_fps, actual_fps)
        except Exception:
            pass

    def set_gain_boost(self, enabled: bool) -> None:
        try:
            if enabled:
                ueye.is_SetGainBoost(self.hcam, ueye.IS_SET_GAINBOOST_ON)
            else:
                ueye.is_SetGainBoost(self.hcam, ueye.IS_SET_GAINBOOST_OFF)
        except Exception:
            pass

    def reinit_aoi(self, width: int, height: int,
                   start_x: int, start_y: int) -> None:
        """
        Change the AOI at runtime. Requires memory reallocation.

        Uses absolute start_x / start_y (same as uEye Cockpit .ini).
        """
        if not self.cfg.use_freeze:
            try:
                ueye.is_StopLiveVideo(self.hcam, ueye.IS_FORCE_VIDEO_STOP)
            except Exception:
                pass

        self._free_memory()

        self._setup_aoi(width, height, use_offsets=False,
                        start_x=start_x, start_y=start_y)

        self._alloc_memory()

        if not self.cfg.use_freeze:
            ret = ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)
            if ret != 0:
                raise RuntimeError(f"is_CaptureVideo restart failed: {ret}")

    # ------------------------------------------------------------------
    # Range queries
    # ------------------------------------------------------------------

    def get_camera_info(self) -> Dict[str, Any]:
        """
        Query all current camera parameters and valid ranges.
        Returns a dict suitable for populating UI controls.
        """
        info: Dict[str, Any] = {}

        info["sensor_width"] = self._sensor_width
        info["sensor_height"] = self._sensor_height

        # --- Pixel clock ---
        try:
            pclk_cur = ueye.uint(0)
            ueye.is_PixelClock(self.hcam, ueye.IS_PIXELCLOCK_CMD_GET, pclk_cur, ueye.sizeof(pclk_cur))
            info["pixel_clock"] = int(pclk_cur)
        except Exception:
            info["pixel_clock"] = 0

        try:
            n_pclk = ueye.uint(0)
            ueye.is_PixelClock(self.hcam, ueye.IS_PIXELCLOCK_CMD_GET_NUMBER, n_pclk, ueye.sizeof(n_pclk))
            pclk_list = (ueye.uint * int(n_pclk))()
            ueye.is_PixelClock(self.hcam, ueye.IS_PIXELCLOCK_CMD_GET_LIST,
                               pclk_list, int(n_pclk) * ueye.sizeof(ueye.uint()))
            info["pixel_clocks"] = sorted(set(int(p) for p in pclk_list))
        except Exception:
            info["pixel_clocks"] = [info.get("pixel_clock", 10)]

        # --- Exposure ---
        try:
            exp_min = ueye.double()
            exp_max = ueye.double()
            exp_inc = ueye.double()
            exp_cur = ueye.double()
            ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, exp_min, ueye.sizeof(exp_min))
            ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, exp_max, ueye.sizeof(exp_max))
            ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, exp_inc, ueye.sizeof(exp_inc))
            ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exp_cur, ueye.sizeof(exp_cur))
            info["exposure_min"] = float(exp_min)
            info["exposure_max"] = float(exp_max)
            info["exposure_inc"] = float(exp_inc)
            info["exposure"] = float(exp_cur)
        except Exception:
            info.update({"exposure_min": 0.01, "exposure_max": 1000.0,
                         "exposure_inc": 0.01, "exposure": 30.0})

        # --- FPS range ---
        try:
            min_time = ueye.double()
            max_time = ueye.double()
            inc_time = ueye.double()
            ueye.is_GetFrameTimeRange(self.hcam, min_time, max_time, inc_time)
            info["fps_min"] = round(1.0 / float(max_time), 2) if float(max_time) > 0 else 0.1
            info["fps_max"] = round(1.0 / float(min_time), 2) if float(min_time) > 0 else 100.0
        except Exception:
            info.update({"fps_min": 0.1, "fps_max": 100.0})

        # --- Gain ---
        info["gain_min"] = 0
        info["gain_max"] = 100
        try:
            cur_gain = ueye.is_SetHardwareGain(self.hcam, ueye.IS_GET_MASTER_GAIN,
                                                ueye.IS_IGNORE_PARAMETER,
                                                ueye.IS_IGNORE_PARAMETER,
                                                ueye.IS_IGNORE_PARAMETER)
            info["gain"] = int(cur_gain)
        except Exception:
            info["gain"] = 0

        # --- Gain boost ---
        try:
            boost = ueye.is_SetGainBoost(self.hcam, ueye.IS_GET_GAINBOOST)
            info["gain_boost"] = bool(boost)
        except Exception:
            info["gain_boost"] = False

        # --- Gamma ---
        try:
            gamma_val = ueye.uint(0)
            ueye.is_Gamma(self.hcam, ueye.IS_GAMMA_CMD_GET, gamma_val, ueye.sizeof(gamma_val))
            info["gamma"] = float(gamma_val) / 100.0
        except Exception:
            info["gamma"] = 1.0
        info["gamma_min"] = 0.01
        info["gamma_max"] = 10.0

        # --- AOI (read back actual values) ---
        try:
            rect = ueye.IS_RECT()
            ueye.is_AOI(self.hcam, ueye.IS_AOI_IMAGE_GET_AOI, rect, ueye.sizeof(rect))
            info["aoi_x"] = int(rect.s32X)
            info["aoi_y"] = int(rect.s32Y)
            info["aoi_width"] = int(rect.s32Width)
            info["aoi_height"] = int(rect.s32Height)
        except Exception:
            info.update({"aoi_x": 0, "aoi_y": 0,
                         "aoi_width": self._live_width, "aoi_height": self._live_height})

        return info

    # ------------------------------------------------------------------
    # Frame grab
    # ------------------------------------------------------------------

    def grab(self) -> np.ndarray:
        if not self._opened:
            raise RuntimeError("Camera not opened")

        if self.cfg.use_freeze:
            ret = ueye.is_FreezeVideo(self.hcam, ueye.IS_WAIT)
            if ret != 0:
                raise RuntimeError(f"is_FreezeVideo failed: {ret}")

        img = ueye.get_data(
            self.mem_ptr,
            self._live_width,
            self._live_height,
            int(self.bitspixel),
            int(self._lineinc),
            copy=True
        )
        
        frame = np.reshape(img, (self._live_height, self._live_width))

        if self.cfg.emit_rgb:
            frame = np.dstack((frame, frame, frame))

        return frame

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if not self._opened:
            return
        try:
            if not self.cfg.use_freeze:
                ueye.is_StopLiveVideo(self.hcam, ueye.IS_FORCE_VIDEO_STOP)
        except Exception:
            pass
        self._free_memory()
        try:
            ueye.is_ExitCamera(self.hcam)
        except Exception:
            pass
        self._opened = False


# =====================================================================
# Camera thread with unified pending-parameter pattern
# =====================================================================

class UEyeCameraThread(QtCore.QThread):
    new_frame = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    # Emitted after camera opens and after any parameter change that affects ranges
    camera_info_signal = QtCore.pyqtSignal(dict)

    def __init__(self, cfg: Optional[UEyeConfig] = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg or UEyeConfig()
        self._cam: Optional[UEyeCamera] = None
        self._running = False

        # Unified pending-parameter dict (thread-safe via mutex)
        self._params_lock = QtCore.QMutex()
        self._pending: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Thread-safe parameter slots (called from UI thread)
    # ------------------------------------------------------------------

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

    def request_info_refresh(self) -> None:
        """Request that the thread emit a fresh camera_info_signal."""
        with QtCore.QMutexLocker(self._params_lock):
            self._pending["refresh_info"] = True

    def stop(self) -> None:
        self._running = False
        self.requestInterruption()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        self._cam = UEyeCamera(self.cfg)
        try:
            self._cam.open()
            self.status.emit("uEye camera opened.")
        except Exception as e:
            self.error.emit(f"Camera open failed: {e}")
            return

        # Emit initial camera info so UI can populate ranges
        try:
            self.camera_info_signal.emit(self._cam.get_camera_info())
        except Exception:
            pass

        dt_min = 1.0 / max(float(self.cfg.max_fps), 0.1)
        t_last = 0.0

        try:
            while self._running and not self.isInterruptionRequested():
                # ---- Apply all pending parameter changes ----
                with QtCore.QMutexLocker(self._params_lock):
                    pending = dict(self._pending)
                    self._pending.clear()

                need_info_update = False

                # Pixel clock first (changes other ranges)
                if "pixel_clock" in pending:
                    try:
                        self._cam.set_pixel_clock(pending["pixel_clock"])
                        need_info_update = True
                        self.status.emit(f"Pixel clock set to {pending['pixel_clock']} MHz")
                    except Exception as e:
                        self.error.emit(f"Pixel clock set failed: {e}")

                # AOI change (heavy: memory realloc)
                if "aoi" in pending:
                    w, h, sx, sy = pending["aoi"]
                    try:
                        self._cam.reinit_aoi(w, h, sx, sy)
                        need_info_update = True
                        self.status.emit(f"AOI: {self._cam.live_width}x{self._cam.live_height} at ({sx},{sy})")
                    except Exception as e:
                        self.error.emit(f"AOI change failed: {e}")

                # Light params
                if "gain" in pending:
                    try:
                        self._cam.set_master_gain(pending["gain"])
                    except Exception as e:
                        self.error.emit(f"Gain set failed: {e}")

                if "gain_boost" in pending:
                    try:
                        self._cam.set_gain_boost(pending["gain_boost"])
                    except Exception as e:
                        self.error.emit(f"Gain boost set failed: {e}")

                if "gamma" in pending:
                    try:
                        self._cam.set_gamma(pending["gamma"])
                    except Exception as e:
                        self.error.emit(f"Gamma set failed: {e}")

                if "exposure" in pending:
                    try:
                        self._cam.set_exposure_ms(pending["exposure"])
                    except Exception as e:
                        self.error.emit(f"Exposure set failed: {e}")

                if "refresh_info" in pending:
                    need_info_update = True

                # Emit updated ranges after any structural change
                if need_info_update:
                    try:
                        self.camera_info_signal.emit(self._cam.get_camera_info())
                    except Exception:
                        pass

                # ---- Grab frame ----
                try:
                    frame = self._cam.grab()
                except Exception as e:
                    self.error.emit(f"Frame grab failed: {e}")
                    continue

                self.new_frame.emit(frame)

                # ---- Soft throttle ----
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
            self.status.emit("uEye camera closed.")


# =====================================================================
# uEye Cockpit .ini loader
# =====================================================================

def load_ueye_config_from_ini(ini_path: str, **overrides) -> UEyeConfig:
    """
    Parse a uEye Cockpit-exported .ini file and return a UEyeConfig.

    The .ini contains [Image size], [Timing], [Gain], [Parameters], etc.
    Fields not present in the .ini fall back to UEyeConfig defaults.

    Any keyword argument in `overrides` takes final precedence.
    """
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

    pclk = _getint("Timing", "Pixelclock", 10)
    kwargs["pixel_clock_mhz"] = pclk

    fps = _getfloat("Timing", "Framerate", 0)
    if fps > 0:
        kwargs["target_fps"] = fps
        kwargs["max_fps"] = fps + 5.0

    kwargs["exposure_ms"] = _getfloat("Timing", "Exposure", 30.0)

    kwargs["master_gain"] = _getint("Gain", "Master", 10)
    kwargs["enable_gain_boost"] = bool(_getint("Gain", "GainBoost", 0))

    gamma = _getfloat("Parameters", "Gamma", 1.0)
    if gamma > 0:
        kwargs["gamma"] = gamma

    kwargs["roi_offset_x"] = 0
    kwargs["roi_offset_y"] = 0

    kwargs.update(overrides)
    return UEyeConfig(**kwargs)


def apply_ini_to_camera(cam: "UEyeCamera", ini_path: str) -> None:
    """
    Apply additional .ini settings beyond UEyeConfig fields
    (hotpixel correction, hardware gamma, and the .ini's exact AOI position).

    Call AFTER cam.open().
    """
    import configparser

    cp = configparser.ConfigParser()
    cp.read(ini_path, encoding="utf-8-sig")

    def _getint(section, key, fallback=0):
        try:
            return int(cp.get(section, key))
        except Exception:
            return fallback

    # Hotpixel correction
    hp_mode = _getint("Parameters", "Hotpixel Mode", -1)
    if hp_mode >= 0:
        try:
            n_mode = ueye.int(hp_mode)
            ueye.is_HotPixel(cam.hcam, ueye.IS_HOTPIXEL_ENABLE_CAMERA_CORRECTION, n_mode, ueye.sizeof(n_mode))
        except Exception:
            pass

    # Hardware gamma
    hw_gamma = _getint("Parameters", "Hardware Gamma", 0)
    if hw_gamma:
        try:
            ueye.is_SetHardwareGamma(cam.hcam, ueye.IS_SET_HW_GAMMA_ON)
        except Exception:
            pass

    # Apply .ini's exact AOI start position
    try:
        ini_sx = _getint("Image size", "Start X", -1)
        ini_sy = _getint("Image size", "Start Y", -1)
        ini_w = _getint("Image size", "Width", -1)
        ini_h = _getint("Image size", "Height", -1)
        if ini_sx >= 0 and ini_sy >= 0 and ini_w > 0 and ini_h > 0:
            cam.reinit_aoi(ini_w, ini_h, ini_sx, ini_sy)
    except Exception:
        pass


def save_settings_to_ini(ini_path: str, settings: dict) -> None:
    """
    Write camera + display settings to a uEye Cockpit-compatible .ini file.

    `settings` is the dict from CameraSettingsDock.get_current_settings():
        pixel_clock, exposure, gain, gain_boost, gamma,
        aoi_width, aoi_height, aoi_x, aoi_y,
        rotation_k, flip_x, flip_y

    The resulting .ini can be loaded by load_ueye_config_from_ini() on next launch,
    AND can be opened in uEye Cockpit for the camera-hardware fields.

    Display-only settings (rotation, flip) are stored in an extra [Display] section
    that uEye Cockpit will ignore but our loader will pick up.
    """
    import configparser

    # If the file already exists, read it first to preserve sections we don't touch
    cp = configparser.ConfigParser()
    try:
        cp.read(ini_path, encoding="utf-8-sig")
    except Exception:
        pass  # fresh file

    # Ensure required sections exist
    for section in ("Image size", "Timing", "Gain", "Parameters", "Display"):
        if not cp.has_section(section):
            cp.add_section(section)

    # Image size
    cp.set("Image size", "Width", str(settings.get("aoi_width", 1280)))
    cp.set("Image size", "Height", str(settings.get("aoi_height", 1024)))
    cp.set("Image size", "Start X", str(settings.get("aoi_x", 0)))
    cp.set("Image size", "Start Y", str(settings.get("aoi_y", 0)))

    # Timing
    cp.set("Timing", "Pixelclock", str(settings.get("pixel_clock", 10)))
    exposure = settings.get("exposure", 30.0)
    cp.set("Timing", "Exposure", f"{exposure:.6f}")
    # Derive framerate from exposure for Cockpit compatibility
    if exposure > 0:
        cp.set("Timing", "Framerate", f"{1000.0 / exposure:.6f}")

    # Gain
    cp.set("Gain", "Master", str(settings.get("gain", 0)))
    cp.set("Gain", "GainBoost", str(int(settings.get("gain_boost", False))))

    # Parameters
    cp.set("Parameters", "Gamma", f"{settings.get('gamma', 1.0):.6f}")

    # Display (our custom section — ignored by uEye Cockpit, read by our loader)
    k = settings.get("rotation_k", 0)
    cp.set("Display", "rotation_k", str(k))
    cp.set("Display", "flip_x", str(int(settings.get("flip_x", False))))
    cp.set("Display", "flip_y", str(int(settings.get("flip_y", False))))

    with open(ini_path, "w", encoding="utf-8") as f:
        cp.write(f)


def _load_display_settings_from_ini(ini_path: str) -> dict:
    """
    Read the [Display] section from a saved .ini (rotation_k, flip_x, flip_y).
    Returns empty dict if section is missing (e.g. pure Cockpit export).
    """
    import configparser
    cp = configparser.ConfigParser()
    try:
        cp.read(ini_path, encoding="utf-8-sig")
    except Exception:
        return {}

    out = {}
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


# Standalone demo
def _standalone() -> None:
    try:
        import cv2
    except Exception:
        raise ImportError("OpenCV (cv2) is required for standalone camera preview")

    cfg = UEyeConfig()
    cam = UEyeCamera(cfg)
    cam.open()

    print("Camera info:", cam.get_camera_info())

    exp = cfg.exposure_ms
    prev = time.time()
    try:
        while True:
            frame = cam.grab()
            if frame.ndim == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                bgr = frame[:, :, ::-1]

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-9)
            prev = now

            label = f"FPS: {int(fps)} | Exp: {exp:.1f} ms"
            cv2.putText(bgr, label, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv2.imshow("uEye Camera", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                exp += 1.0
                cam.set_exposure_ms(exp)
            elif key == ord("s"):
                exp = max(0.1, exp - 1.0)
                cam.set_exposure_ms(exp)
    finally:
        cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    _standalone()
