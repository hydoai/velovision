'''
Interface for camera-specific, NVIDIA-hardware accelerated camera access and video recording.

Use example:

    cap = cameras.pi_v2_cam(width=1280, height=720) # cap is a cv2.VideoCapture object

    while True:
        ret_val, frame = cap.read()
        if ret_val == True:
            # do image processing
'''

import cv2
import os

def create_cv_cap_piv2()

def create_cv_cap_imx291()

def clone_stream()

def record_stream_piv2()

def record_stream_imx291()


if __name__ == '__main__':
    # testing
    print("Make sure that Raspberry Pi V2 camera is connected to CSI")
    print("Make sure that IMX291 USB camera is plugged in")

    cap0 = create_cv_cap_pi()


