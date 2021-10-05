# How to set up duplicated video pipeline for saving & inference


## Documentation

### V4L2loopback
https://github.com/umlaeute/v4l2loopback
https://github.com/umlaeute/v4l2loopback/wiki

### NVIDIA Gstreamer plugin
https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html#wwpID0E0R40HA

## Installation

**Install V4L2loopback**
```bash
sudo apt install v4l2loopback-dkms
```

**Install NVIDIA Gstreamer plugin**
```bash
sudo add-apt-repository universe
sudo add-apt-repository multiverse
sudo apt-get update
sudo apt-get install gstreamer1.0-tools gstreamer1.0-alsa \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
sudo apt-get install libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-good1.0-dev \
  libgstreamer-plugins-bad1.0-dev
```

## Check Camera Specification (capability)

List possible formats for `/dev/video0`:
```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## Run

Add four new virtual /dev/video* devices.
```bash
sudo modprobe v4l2loopback devices=4
```

See that new video/dev* devices have been created:
```bash
ls /dev/video*
```



