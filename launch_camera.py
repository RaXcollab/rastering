from pyueye import ueye
import numpy as np
import cv2
# simple example to open camera, set some parameters, take live video, and close the camera.

def main():
    # initialize the camera.
    hcam = ueye.HIDS(0)
    ret = ueye.is_InitCamera(hcam, None)
    print(f"initCamera returns {ret}")

    # set the color mode.
    ret = ueye.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)
    print(f"SetColorMode IS_CM_BGR8_PACKED returns {ret}")

    # set the region of interest (Camera dependent).
    width = 1280
    height = 1024
    rect_aoi = ueye.IS_RECT()
    rect_aoi.s32X = ueye.int(0)
    rect_aoi.s32Y = ueye.int(0)
    rect_aoi.s32Width = ueye.int(width)
    rect_aoi.s32Height = ueye.int(height)
    ueye.is_AOI(hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
    print(f"AOI IS_AOI_IMAGE_SET_AOI returns {ret}")

    # allocate memory for live view.
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.int()
    bitspixel = 24 # for colormode = IS_CM_BGR8_PACKED
    ret = ueye.is_AllocImageMem(hcam, width, height, bitspixel,
                                mem_ptr, mem_id)
    print(f"AllocImageMem returns {ret}")
    
    # set active memory region.
    ret = ueye.is_SetImageMem(hcam, mem_ptr, mem_id)
    print(f"SetImageMem returns {ret}")

    # continuous capture to memory.
    ret = ueye.is_CaptureVideo(hcam, ueye.IS_DONT_WAIT)
    print(f"CaptureVideo returns {ret}")
    
    # get data from camera and display (press the q key to exit out of live capture).
    lineinc = width * int((bitspixel + 7) / 8)

    # set exposure
    time_exposure_ = 3
    time_exposure = ueye.double(time_exposure_)
    ret = ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))

    while True:
        img = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)
        img = np.reshape(img, (height, width, 3))
        cv2.imshow('uEye Python Example (a,s=exposure,q to exit)', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('a'):
            time_exposure_ += 0.1
            time_exposure = ueye.double(time_exposure_)
            ret = ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))
        elif key == ord('s'):
            time_exposure_ -= 0.1
            time_exposure = ueye.double(time_exposure_)
            ret = ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))

    cv2.destroyAllWindows()
    
    # stop capture and close the camera.
    ret = ueye.is_StopLiveVideo(hcam, ueye.IS_FORCE_VIDEO_STOP)
    print(f"StopLiveVideo returns {ret}")
    ret = ueye.is_ExitCamera(hcam)
    print(f"ExitCamera returns {ret}")

    print("img shape=", img.shape)

    image_array = img
    if image_array.ndim == 3 and image_array.shape[2] == 3:  # RGB
        h, w, c = image_array.shape
        q_image = QImage(image_array.data, w, h, 3 * w, QImage.Format_RGB888)
    elif image_array.ndim == 2:  # Grayscale image
        h, w = image_array.shape
    # Convert NumPy array (uint8) to QImage (grayscale)
        q_image = QImage(image_array.data, w, h, w, QImage.Format_Grayscale8)
    elif image_array.ndim == 3 and image_array.shape[2] == 4:  # RGBA
        h, w, c = image_array.shape
        q_image = QImage(image_array.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return q_image

if __name__ == '__main__':
    main()