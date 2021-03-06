'''
Interface for camera-specific, NVIDIA-hardware accelerated camera access and video recording.

'''
import time
import datetime
from time import sleep
import os
from multiprocessing import Process
import subprocess
import cv2
import getpass

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
    def __init__(self):
        self.num_virtual_devices = 4
        self.max_buffers = 2
        username = getpass.getuser()
        self.video_save_path = f'/media/{username}/microsdcard' #'/home/halpert/Videos/'

        sources = (4,6) # which dev/video numbers will be used for black box
        for source in sources:
            save_dir = os.path.join(self.video_save_path, f'video_stream_{source}')
            os.makedirs(save_dir, exist_ok=True)
        

    def start_pipelines(self):
        # create loopback (virtual) /dev/video* devices
        modprobe_command = f'sudo modprobe v4l2loopback devices={self.num_virtual_devices} max_buffers={self.max_buffers}'
        os.system(modprobe_command) # PLEASE NOTE: sudoers file must be edited to allow user to run sudo commands. Please see gi-dev-setup for more info

        # create first set of loopbacks (access to /dev/video0 and /dev/video1 is exclusive, but loopback devices can be multiple acceessed)
        # also, the incantation is specific to camera type
        self.first_clone_csi(source=0, dest=3)
        self.first_clone_usb(source=1, dest=5)

        sleep(3) #IMPORTANT; pipelines need time to set up

        # second set of loopbacks (which are cloning loopback devices, so it can be accessed by any number of consumers)
        self.clone_loopback(source=3, dest=4) # both source and dest index devices are available for use, but stopping an access to source will kill the pipeline
        self.clone_loopback(source=5, dest=6)

        self.clear_disk_space()

    def create_cap(self,index):
        return cv2.VideoCapture(index)

    def clone_loopback(self, source, dest):
        def clone(source, dest):
            os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! v4l2sink device=/dev/video{dest}")
            sleep(2)
        proc = Process(target=clone, args=([source, dest]))
        proc.start()

    def first_clone_csi(self, source, dest):
        def csi_loopback(source, dest):
            width = 1280 
            height = 720 
            fps = 60
            os.system(f"gst-launch-1.0 nvarguscamerasrc sensor-id={source} ! 'video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1, format=I420' ! nvvidconv output-buffers=4 ! 'video/x-raw, width={width}, height={height}, framerate={fps}/1, format=UYVY' ! identity drop-allocation=true ! v4l2sink device=/dev/video{dest}")
            sleep(2)

        proc = Process(target=csi_loopback, args=([source,dest]))
        proc.start()

    def first_clone_usb(self, source, dest):
        def usb_loopback(source, dest):
            os.system(f"gst-launch-1.0 v4l2src device=/dev/video{source} ! v4l2sink device=/dev/video{dest}")
            sleep(2)

        proc = Process(target=usb_loopback, args=([source,dest]))
        proc.start()

    def record_nvenc_h265(self,source, width, height, max_length=60, fps=30):
        '''
        Args:
            max_length: save file and create new file after max_length seconds
        '''


        output_path = os.path.join(self.video_save_path , f'video_stream_{source}')

        def record(source, width, height, output_path, max_length, fps):
            #while True:
            output_path = os.path.join(output_path,f"{time.strftime('%Y-%m-%d--%H-%M-%S')}.mkv")
            command = f"gst-launch-1.0 v4l2src io-mode=2 device=/dev/video{source} ! 'video/x-raw,width={width}, height={height}, framerate={fps}/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h265enc maxperf-enable=1 bitrate=2000000 ! h265parse ! matroskamux ! filesink append=false location={output_path} -e"
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=-1, shell=True)
            # hack to fix bug where sometimes it writes empty files
            sleep(1)
            try:
                size_of_file = os.path.getsize(output_path)
            except OSError:
                size_of_file = 0

            if size_of_file == 0:
                #continue
                pass
            # === end hack

            #sleep(max_length-1) # because we already slept one sec above
            #process.send_signal(signal.SIGINT)
        proc = Process(target=record, args=([source, width, height, output_path, max_length, fps]))
        proc.start()

    def clear_disk_space(self, remaining_threshold_gb=20, interval=300):
        '''
        Delete the most recent files in 'output_dir' every 'interval' seconds so that the remaining storage is greater than 'remaining_threshold_gb'

        Also, clear any empty files (result of a bug I can't seem to fix)
        '''

        def get_remaining_space():
            disk_stats = os.statvfs('/')
            bytes_available =(disk_stats.f_bavail * disk_stats.f_frsize)
            gb_available = bytes_available / 1024 / 1024 / 1024
            return gb_available

        def clear_space(output_dir, remaining_threshold_gb, interval):
            for child_dir in os.listdir(output_dir):
                child_dir_path = os.path.join(output_dir, child_dir)

            while True:
                sleep(interval)
                list_of_video_files = sorted(os.listdir(child_dir_path))
                for filename in list_of_video_files:
                    file_path = os.path.join(child_dir_path, filename)
                    if os.path.getsize(file_path) == 0:
                        os.remove(file_path)

                if get_remaining_space() < remaining_threshold_gb:
                    oldest_files = list_of_video_files[max(10, len(list_of_video_files)):]
                    for file_to_delete in oldest_files:
                        file_path = os.path.join(child_dir_path, file_to_delete)
                        os.remove(file_path)
        proc = Process(target=clear_space, args=([self.video_save_path, remaining_threshold_gb, interval]))
        proc.start()
if __name__ == '__main__':
    # testing
    import argparse
    import time
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    camint = CameraInterface()
    camint.start_pipelines()

    #test_usb_stream(4)
    #test_usb_stream(6)

    camint.record_nvenc_h265(4, 1280, 720, fps=30, max_length=60) # video is saved to /home/dwight/Videos/recording_4.mp4
    camint.record_nvenc_h265(6, 640, 480, fps=30, max_length=60)
    
    #cap3 = camint.create_cap(3)
    #cap5 = camint.create_cap(5)

    while False and cap3.isOpened() and cap5.isOpened():
        ret3, frame3 = cap3.read()
        ret5, frame5 = cap5.read()

        print(frame3.shape)
        print(frame5.shape)




