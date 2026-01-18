from pyueye import ueye
import numpy as np
import cv2
import time

def main():
    # 1. Initialize Camera
    hcam = ueye.HIDS(0)
    ret = ueye.is_InitCamera(hcam, None)
    if ret != 0:
        print(f"InitCamera failed with error {ret}")
        return

    # 2. Set Color Mode
    ueye.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)

    # 3. Set Pixel Clock (Confirmed Stable at 10 MHz)
    pclk = ueye.int(10)
    ueye.is_PixelClock(hcam, ueye.IS_PIXELCLOCK_CMD_SET, pclk, ueye.sizeof(pclk))
    
    # 4. Set Region of Interest (AOI)
    width = 1280
    height = 1024
    rect_aoi = ueye.IS_RECT()
    rect_aoi.s32X = ueye.int(0)
    rect_aoi.s32Y = ueye.int(0)
    rect_aoi.s32Width = ueye.int(width)
    rect_aoi.s32Height = ueye.int(height)
    ueye.is_AOI(hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

    # 5. Allocate Memory
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.int()
    bitspixel = 24
    ueye.is_AllocImageMem(hcam, width, height, bitspixel, mem_ptr, mem_id)
    ueye.is_SetImageMem(hcam, mem_ptr, mem_id)

    # 6. Set Initial Exposure (30ms)
    current_exposure = 30.0
    ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.double(current_exposure), 8)

    # Data extraction variables
    lineinc = width * int((bitspixel + 7) / 8)
    prev_time = 0

    print(f"Camera running at {pclk.value} MHz. Press 'q' to exit, 'a/s' for exposure.")

    while True:
        # Snap one frame (Synchronous) - Prevents static buffer issues
        ret = ueye.is_FreezeVideo(hcam, ueye.IS_WAIT)
        
        if ret != 0:
            print(f"Frame Error: {ret}")
            continue

        # Extract data
        img = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
        img = np.reshape(img, (height, width, 3))

        # --- FPS Calculation ---
        curr_time = time.time()
        try:
            fps = 1 / (curr_time - prev_time)
        except ZeroDivisionError:
            fps = 0
        prev_time = curr_time

        # --- Overlay Info ---
        label = f"FPS: {int(fps)} | Exp: {current_exposure:.1f}ms"
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('uEye Camera', img)

        # --- Controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            current_exposure += 1.0
            ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.double(current_exposure), 8)
        elif key == ord('s'):
            current_exposure = max(0.1, current_exposure - 1.0)
            ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.double(current_exposure), 8)

    # Cleanup
    cv2.destroyAllWindows()
    ueye.is_FreeImageMem(hcam, mem_ptr, mem_id)
    ueye.is_ExitCamera(hcam)

if __name__ == '__main__':
    main()