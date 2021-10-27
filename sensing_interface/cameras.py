'''
Interface for camera-specific, NVIDIA-hardware accelerated camera access and video recording.

Use example:

    cap = cameras.pi_v2_cam(width=1280, height=720) # cap is a cv2.VideoCapture object

    while True:
        ret_val, frame = cap.read()
        if ret_val == True:
            # do image processing
'''
import time
from time import sleep
import os
from multiprocessing import Process
import subprocess
import cv2

# try
import subprocess
import signal

def test_usb_stream(index):
    def view_cam(index):
        os.system(f"gst-launch-1.0 v4l2src device=/dev/video{index} ! xvimagesink")
    proc = Process(target=view_cam, args=([index]))
    proc.start()
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

        sleep(3) #IMPORTANT; pipelines need time to set up

        # second set of loopbacks (which are cloning loopback devices, so it can be accessed by any number of consumers)
        self.clone_loopback(source=3, dest=4) # both source and dest index devices are available for use, but stopping an access to source will kill the pipeline
        self.clone_loopback(source=5, dest=6)

    def create_cap(self,index):
        return cv2.VideoCapture(index)

    def clone_loopback(self, source, dest):
        def clone(source, dest):
            os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! v4l2sink device=/dev/video{dest}")
        proc = Process(target=clone, args=([source, dest]))
        proc.start()

    def first_clone_csi(self, source, dest):
        def csi_loopback(source, dest):
            os.system(f"gst-launch-1.0 nvarguscamerasrc sensor-id={source} ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1, format=I420' ! nvvidconv output-buffers=4 ! 'video/x-raw, width=1280, height=720, framerate=60/1, format=UYVY' ! identity drop-allocation=true ! v4l2sink device=/dev/video{dest}")

        proc = Process(target=csi_loopback, args=([source,dest]))
        proc.start()

    def first_clone_usb(self, source, dest):
        def usb_loopback(source, dest):
            os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! v4l2sink device=/dev/video{dest}")

        proc = Process(target=usb_loopback, args=([source,dest]))
        proc.start()

    def record_nvenc_h264(self,source, width, height, output_path=''):
        output_path = f'/home/dwight/Videos/recording_{source}.mp4'
        def record(source, width, height, output_path):
            os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! 'video/x-raw,width={width}, height={height}, framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink location={output_path} -e")
        proc = Process(target=record, args=([source, width, height, output_path]))
        proc.start()

    def split_record_nvenc_h264(self,source, width, height, output_path='', max_length=10):
        '''
        Args:
            max_length: save file and create new file after max_length seconds
        '''


        def record(source, width, height, output_path):
            #os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! 'video/x-raw,width={width}, height={height}, framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink location={output_path} -e")

            while True:
                output_path = f'/home/dwight/Videos/recording_{source}_{round(time.time())}.mp4'
                command = f"gst-launch-1.0 v4l2src device=/dev/video{source} ! 'video/x-raw,width={width}, height={height}, framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink location={output_path} -e"
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, shell=True)
                sleep(5)
                process.send_signal(signal.SIGINT)
        proc = Process(target=record, args=([source, width, height, output_path]))
        proc.start()

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

    #test_usb_stream(4)
    #test_usb_stream(6)
    camint.split_record_nvenc_h264(4, 1280, 720) # video is saved to /home/dwight/Videos/recording_4.mp4
    camint.split_record_nvenc_h264(6, 640, 480)
    
    sleep(3)

    cap3 = camint.create_cap(3)
    cap5 = camint.create_cap(5)

    while cap3.isOpened() and cap5.isOpened():
        ret3, frame3 = cap3.read()
        ret5, frame5 = cap5.read()

        print(frame3.shape)
        print(frame5.shape)



