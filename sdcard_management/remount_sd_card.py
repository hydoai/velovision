import os
import getpass

def get_username():
    return getpass.getuser()

device_path = '/dev/mmcblk1p1'
mount_path = f'/media/{get_username()}/microsdcard'

def unmount_sd_card(device_path):
    umount_command = f'sudo umount {device_path}'
    os.system(umount_command)

def mount_sd_card(mount_path):
    # create a mount point
    create_mount_point_command = f'sudo mkdir {mount_path}'
    os.system(create_mount_point_command)

    # mount device to mount point
    mount_command = f'sudo mount {device_path} {mount_path}'
    os.system(mount_command)

def remount_sd_card(device_path, mount_path):
    unmount_sd_card(device_path)
    mount_sd_card(mount_path)

if __name__ == '__main__':

    print(f"User name is: {get_username()}")
    print("unmounting sd card (/dev/mmcblk1p1)")
    unmount_sd_card(device_path)
    print(f"remounting sd card at /media/{get_username()}/microsdcard")
    mount_sd_card(mount_path)

