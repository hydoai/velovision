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
import subprocess

class CameraInterface:
    def __init__(self, sudo_password):
        self.sudo_password = sudo_password
        
        self.num_virtual_devices = 4
        self.max_buffers = 2
        modprobe_command = f'sudo modprobe v4l2loopback devices={self.num_virtual_devices} max_buffers=2{self.max_buffers}'
        os.system('echo %s|sudo -S %s' % (self.sudo_password, modprobe_command))

        

def create_cv_cap_piv2():
    pass

def create_cv_cap_imx291():
    pass

def clone_stream():
    pass

def record_stream_piv2():
    pass

def record_stream_imx291():
    pass


if __name__ == '__main__':
    # testing
    print("Make sure that Raspberry Pi V2 camera is connected to CSI")
    print("Make sure that IMX291 USB camera is plugged in")
    
    camint = CameraInterface()
