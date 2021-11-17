import os 
from pathlib import Path
import glob
import math
import time
from time import sleep
from threading import Thread

from remount_sd_card import get_username, device_path, mount_path

video_suffix = ['.mp4', '.mkv', '.avi', '.m4v']

def get_remaining_space(path):
    disk_stats = os.statvfs(path)
    bytes_available = (disk_stats.f_bavail * disk_stats.f_frsize)
    gb_available = bytes_available / 1024 / 1024 / 1024
    return gb_available

def fill_to_nearest_gb(path):
    '''
    Debugging function to fill a given drive to near full
    '''
    free_space = get_remaining_space(path)
    floor_space = int(math.floor(free_space))

    print()
    print(f"Creating random files to fill {floor_space} GB...")

    for i in range(floor_space):
        filename = f"{time.time()}-random.mkv"
        pathname = os.path.join(path, filename)
        random_fill_command = f'dd if=/dev/urandom of={pathname} bs=1G iflag=fullblock count=1'
        os.system(random_fill_command)



def free_space(path, min_space_remaining, check_interval):
    '''
    path: Path to mounted drive (usually in /media/username...)
    min_space_remaining: delete as many old files it takes to get 'min_space_remaining' GB of space free.
    check_interval: run this check every _ seconds
    '''
    while True:
        while get_remaining_space(path) < min_space_remaining:
            paths = sorted(Path(path).iterdir(), key=os.path.getmtime)
            oldest_video_paths = [x for x in paths if (x.suffix in video_suffix)]
            if len(oldest_video_paths):
                oldest_path = oldest_video_paths.pop(0)
                print(f"Deleting old file: {oldest_path}")
                os.remove(oldest_path)
        sleep(check_interval)

def threaded_free_space(path, min_space_remaining, check_interval=30):
    '''
    path: Path to mounted drive (usually in /media/username...)
    min_space_remaining: delete as many old files it takes to get 'min_space_remaining' GB of space free.
    check_interval: run this check every _ seconds
    '''
    t = Thread(target=free_space, args=([path, min_space_remaining, check_interval]))
    t.start()

if __name__ == '__main__':
    print(f"Remaining Space on {mount_path}: {get_remaining_space(mount_path)} GB")
    # DEBUGGING
    print()
    print("Filling micro sd card to the nearest gb. This might take a while.")
    print("Note: You may not be able to see the created files until it is all done")
    #fill_to_nearest_gb(mount_path)

    # ACTUAL
    print("Running free_space(), which will delete old files until we have 10GB free")
    threaded_free_space(mount_path, 10)

    print(f"Remaining Space on {mount_path}: {get_remaining_space(mount_path)} GB")


