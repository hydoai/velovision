# How to set up duplicated video pipeline for saving & inference


## Documentation
https://github.com/umlaeute/v4l2loopback
https://github.com/umlaeute/v4l2loopback/wiki

## Installation
```bash
sudo apt install v4l2loopback-dkms
```

## Run

Add four new virtual /dev/video* devices.
```bash
sudo modprobe v4l2loopback devices=4
```



