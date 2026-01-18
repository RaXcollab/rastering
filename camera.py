# camera.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

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
    emit_rgb: bool = True              # UI expects RGB-ish; uEye often gives BGR
    max_fps: float = 30.0              # soft throttle
    roi_offset_x: int = 0
    roi_offset_y: int = 0


class UEyeCamera:
    """
    Thin uEye wrapper with explicit init/close and a single-frame grab.
    """
    def __init__(self, cfg: UEyeConfig):
        self.cfg = cfg
        self.hcam = ueye.HIDS(int(cfg.camera_id))

        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()
        self.bitspixel = 24  # BGR8 packed

        self._lineinc = int(cfg.width) * int((self.bitspixel + 7) / 8)
        self._opened = False

    def open(self) -> None:
        ret = ueye.is_InitCamera(self.hcam, None)
        if ret != 0:
            raise RuntimeError(f"is_InitCamera failed: {ret}")

        # Color mode
        ueye.is_SetColorMode(self.hcam, ueye.IS_CM_BGR8_PACKED)

        # Disable auto shutter + gain
        val = ueye.double(0)
        dummy = ueye.double(0)
        ueye.is_SetAutoParameter(self.hcam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, val, dummy)
        ueye.is_SetAutoParameter(self.hcam, ueye.IS_SET_ENABLE_AUTO_GAIN, val, dummy)
        # -------------------------------------------

        # Pixel clock
        pclk = ueye.int(int(self.cfg.pixel_clock_mhz))
        ueye.is_PixelClock(self.hcam, ueye.IS_PIXELCLOCK_CMD_SET, pclk, ueye.sizeof(pclk))

        # AOI setup
        sinfo = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(self.hcam, sinfo)
        max_w = int(sinfo.nMaxWidth)
        max_h = int(sinfo.nMaxHeight)

        req_w = int(self.cfg.width)
        req_h = int(self.cfg.height)
        
        # Calculate centered offsets
        off_x = (max_w - req_w) // 2
        off_y = (max_h - req_h) // 2
        
        # Apply user offsets
        off_x -= int(self.cfg.roi_offset_x)
        off_y -= int(self.cfg.roi_offset_y)

        # Clamp to sensor bounds
        off_x = max(0, min(off_x, max_w - req_w))
        off_y = max(0, min(off_y, max_h - req_h))

        # Align to 4 pixels (hardware requirement)
        off_x = (off_x // 4) * 4
        off_y = (off_y // 4) * 4

        # Apply AOI
        rect = ueye.IS_RECT()
        rect.s32X = ueye.int(off_x)
        rect.s32Y = ueye.int(off_y)
        rect.s32Width = ueye.int(req_w)
        rect.s32Height = ueye.int(req_h)
        ueye.is_AOI(self.hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect, ueye.sizeof(rect))

        # Allocate + set memory for the AOI size
        ret = ueye.is_AllocImageMem(
            self.hcam,
            ueye.int(req_w),
            ueye.int(req_h),
            ueye.int(self.bitspixel),
            self.mem_ptr,
            self.mem_id
        )
        if ret != 0:
            raise RuntimeError(f"is_AllocImageMem failed: {ret}")

        ret = ueye.is_SetImageMem(self.hcam, self.mem_ptr, self.mem_id)
        if ret != 0:
            raise RuntimeError(f"is_SetImageMem failed: {ret}")

        # If you want capture mode, start it; FreezeVideo does not strictly require CaptureVideo.
        if not self.cfg.use_freeze:
            ret = ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)
            if ret != 0:
                raise RuntimeError(f"is_CaptureVideo failed: {ret}")

        # Exposure
        self.set_exposure_ms(self.cfg.exposure_ms)

        self._opened = True

    def set_exposure_ms(self, exposure_ms: float) -> None:
        exposure_ms = float(exposure_ms)
        exp = ueye.double(exposure_ms)
        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp, ueye.sizeof(exp))
        if ret != 0:
            raise RuntimeError(f"is_Exposure set failed: {ret}")

    def grab(self) -> np.ndarray:
        if not self._opened:
            raise RuntimeError("Camera not opened")

        if self.cfg.use_freeze:
            ret = ueye.is_FreezeVideo(self.hcam, ueye.IS_WAIT)
            if ret != 0:
                raise RuntimeError(f"is_FreezeVideo failed: {ret}")

        img = ueye.get_data(
            self.mem_ptr,
            int(self.cfg.width),
            int(self.cfg.height),
            int(self.bitspixel),
            int(self._lineinc),
            copy=True
        )
        frame = np.reshape(img, (int(self.cfg.height), int(self.cfg.width), 3))

        # uEye gives BGR in IS_CM_BGR8_PACKED; UI typically wants RGB
        if self.cfg.emit_rgb:
            frame = frame[:, :, ::-1].copy()

        return frame

    def close(self) -> None:
        if not self._opened:
            return
        try:
            if not self.cfg.use_freeze:
                ueye.is_StopLiveVideo(self.hcam, ueye.IS_FORCE_VIDEO_STOP)
        except Exception:
            pass
        try:
            ueye.is_FreeImageMem(self.hcam, self.mem_ptr, self.mem_id)
        except Exception:
            pass
        try:
            ueye.is_ExitCamera(self.hcam)
        except Exception:
            pass
        self._opened = False


class UEyeCameraThread(QtCore.QThread):
    new_frame = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, cfg: Optional[UEyeConfig] = None, parent=None):
        super().__init__(parent)
        self.cfg = cfg or UEyeConfig()
        self._cam: Optional[UEyeCamera] = None
        self._running = False

        self._exposure_lock = QtCore.QMutex()
        self._pending_exposure: Optional[float] = None

    @QtCore.pyqtSlot(float)
    def set_exposure_ms(self, exposure_ms: float) -> None:
        # Thread-safe request; applied in run loop
        with QtCore.QMutexLocker(self._exposure_lock):
            self._pending_exposure = float(exposure_ms)

    def stop(self) -> None:
        self._running = False
        self.requestInterruption()

    def run(self) -> None:
        self._running = True
        self._cam = UEyeCamera(self.cfg)
        try:
            self._cam.open()
            self.status.emit("uEye camera opened.")
        except Exception as e:
            self.error.emit(f"Camera open failed: {e}")
            return

        dt_min = 1.0 / max(float(self.cfg.max_fps), 0.1)
        t_last = 0.0

        try:
            while self._running and not self.isInterruptionRequested():
                # apply pending exposure if any
                with QtCore.QMutexLocker(self._exposure_lock):
                    exp = self._pending_exposure
                    self._pending_exposure = None
                if exp is not None:
                    try:
                        self._cam.set_exposure_ms(exp)
                    except Exception as e:
                        self.error.emit(f"Exposure set failed: {e}")

                # grab
                try:
                    frame = self._cam.grab()
                except Exception as e:
                    self.error.emit(f"Frame grab failed: {e}")
                    continue

                self.new_frame.emit(frame)

                # soft throttle
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


# Standalone demo
def _standalone() -> None:
    try:
        import cv2
    except Exception:
        raise ImportError("OpenCV (cv2) is required for standalone camera preview")

    cfg = UEyeConfig()
    cam = UEyeCamera(cfg)
    cam.open()

    exp = cfg.exposure_ms
    prev = time.time()
    try:
        while True:
            frame = cam.grab()  # RGB if emit_rgb=True
            # cv2 expects BGR:
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
