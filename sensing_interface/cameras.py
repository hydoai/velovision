'''
Interface for camera-specific, NVIDIA-hardware accelerated camera access and video recording.

Use example:

    cap = cameras.pi_v2_cam(width=1280, height=720) # cap is a cv2.VideoCapture object

    while True:
        ret_val, frame = cap.read()
        if ret_val == True:
            # do image processing
'''
from time import sleep
import os
from multiprocessing import Process
import subprocess
import cv2

def test_usb_stream(index):
    os.system(f"gst-launch-1.0 v4l2src device=/dev/video{index} ! xvimagesink")
def test_csi_stream(index):
    os.system(f"gst-launch-1.0 nvarguscamerasrc sensor-id={index} ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw, width=960, height=616' ! xvimagesink  -e")
class CameraInterface:
    def __init__(self, sudo_password):
        self.sudo_password = sudo_password
        
        # create loopback (virtual) /dev/video* devices
        self.num_virtual_devices = 4
        self.max_buffers = 2
        modprobe_command = f'sudo modprobe v4l2loopback devices={self.num_virtual_devices} max_buffers={self.max_buffers}'
        os.system('echo %s|sudo -S %s' % (self.sudo_password, modprobe_command))

        # create first set of loopbacks (access to /dev/video0 and /dev/video1 is exclusive, but loopback devices can be multiple acceessed)
        # also, the incantation is specific to camera type
        self.first_clone_csi(source=0, dest=3)
        self.first_clone_usb(source=1, dest=5)

        sleep(3)

        # second set of loopbacks (which are cloning loopback devices, so it can be accessed by any number of consumers)
        self.clone_loopback(source=3, dest=4) # both source and dest index devices are available for use, but stopping an access to source will kill the pipeline
        self.clone_loopback(source=5, dest=6)

    def clone_loopback(self, source, dest):
        def clone(source, dest):
            os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! v4l2sink device=/dev/video{dest}")
        proc = Process(target=clone, args=([source, dest]))
        proc.start()

    def first_clone_csi(self, source, dest):
        def csi_loopback(source, dest):
            os.system(f"gst-launch-1.0 nvarguscamerasrc sensor-id={source} ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=I420' ! nvvidconv output-buffers=4 ! 'video/x-raw, width=1920, height=1080, framerate=30/1, format=UYVY' ! identity drop-allocation=true ! v4l2sink device=/dev/video{dest}")

        proc = Process(target=csi_loopback, args=([source,dest]))
        proc.start()

    def first_clone_usb(self, source, dest):
        def usb_loopback(source, dest):
            os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! v4l2sink device=/dev/video{dest}")

        proc = Process(target=usb_loopback, args=([source,dest]))
        proc.start()

        



        

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
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--password', type=str, help='Root password')
    args = parser.parse_args()
    if not args.password: 
        print("Please enter root password in '--password' argument")
    
    camint = CameraInterface(args.password)

    proc1 = Process(target=test_usb_stream, args=([4]))
    proc2 = Process(target=test_usb_stream, args=([6]))
    proc1.start()
    proc2.start()
