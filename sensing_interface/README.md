# Sensing Interface

## Project Gimondi Specific Setup

```
┌───────────────┐    ┌───────────────┐
│Raspberry Pi V2│    │ Arducam IMX291│
└───────┬───────┘    └──────┬────────┘
        │                   │
        │MIPI/CSI       USB ├───────────────┐
        │                   │               │
  ┌─────▼─────┐       ┌─────▼─────┐   ......▼......
  │/dev/video0│       │/dev/video1│   ./dev/video2.
  └─────┬─────┘       └──────────┬┘   .............
        │                        │        audio
        │                        │
        │                        │
  ┌─────▼─────┐  ┌────────┐  ┌───▼───────┐
  │/dev/video3├─►│Computer│◄─┤/dev/video5│
  └─────┬─────┘  │Vision  │  └─────┬─────┘
        │        └────────┘        │
  ┌─────▼─────┐              ┌─────▼─────┐
  │/dev/video4│              │/dev/video6│
  └─────┬─────┘              └─────┬─────┘
        │                          │
        │                          │
        ▼                          ▼
    video file                 video file
```


# How to set up duplicated video pipeline for saving & inference


## Documentation

### V4L2loopback
https://github.com/umlaeute/v4l2loopback
https://github.com/umlaeute/v4l2loopback/wiki

### NVIDIA Gstreamer plugin
https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html#wwpID0E0R40HA

### How to run GStreamer in Python
http://lifestyletransfer.com/how-to-launch-gstreamer-pipeline-in-python/

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

## Minimal Demo

Assuming USB camera is plugged into `/dev/video1`:

Add two new virtual /dev/video* devices.
```bash
sudo modprobe v4l2loopback devices=4 max_buffers=2
```

See that new video/dev* devices have been created:
```bash
ls /dev/video*
```

Create two new virtual streams:
```bash
gst-launch-1.0 v4l2src device=/dev/video1 ! v4l2sink device=/dev/video2
```
```bash
gst-launch-1.0 v4l2src device=/dev/video2 ! v4l2sink device=/dev/video3
```

View first virtual stream:
```bash
gst-launch-1.0 v4l2src device=/dev/video2 ! xvimagesink
```

View MJPEG stream (instead of RAW; for higher resolution)
```bash
gst-launch-1.0 v4l2src io-mode=2 device=/dev/video1 ! "image/jpeg,framerate=30/1,width=1280,height=720" ! jpegdec ! video/x-raw ! xvimagesink

```
Save video from second virtual stream: (note: very inefficient on Jetson Nano because no hardware acceleration)
```bash
gst-launch-1.0 v4l2src device=/dev/video3 ! videoconvert ! x264enc ! mp4mux ! filesink location=/home/dwight/Videos/test-gst.mp4 -e
```
+ `-e` is important: It sends EOS message to the video file when process is exited. Without it, the video file will not play.

# Camera-Specific Settings & Examples

## USB Camera -> 1st Virtual Stream -> Hardware Accelerated Video Encode (saving)

Assuming USB camera input is at `/dev/video1`.

Create a virtual stream at `/dev/video2`:

```bash
gst-launch-1.0 v4l2src device=/dev/video1 ! v4l2sink device=/dev/video2
```

Convert stream to NVIDIA hardware accceleration format and save to `OUTPUT.mp4`:
```bash
gst-launch-1.0 v4l2src device=/dev/video2 ! 'video/x-raw,width=1280, height=720, framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=I420' ! nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink location=OUTPUT.mp4 -e
```


## 2nd Virtual Stream -> Computer Vision Processing
Create a second virtual stream from `/dev/video2` to `/dev/video3`:
```bash
gst-launch-1.0 v4l2src device=/dev/video2 ! v4l2sink device=/dev/video3
```

**Start jetson-inference session**:
Navigate to jetson-inference detectNet demo:
```bash
cd jetson-inference/aarch/build
```
Start detectNet demo with virtual stream as input:
```bash
./detectnet /dev/video3
```
Currently, this runs Mobilenet-SSD. It has terrible accuracy. Hope to use YOLOX nano.

Alternative: (very slow, 10FPS on 30FPS stream)
Open `/dev/video3` with OpenCV in Python:
```python3
import cv2
import numpy as np

CAM_INDEX = 3

cap = cv2.VideoCapture(CAM_INDEX,cv2.CAP_V4L)
cap.set(6, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(3, 1280)
cap.set(4, 720)
cap.set(5, 30)
print(cap)
print(cap.isOpened())
while cap.isOpened():
    _, frame = cap.read()
    print(cap.get(5))
    print(frame.shape)
```



## Raspberry Pi V2 CSI Camera

### View camera
```bash
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw, width=960, height=616' ! xvimagesink -e
```
### Save video from camera (selected ones from NVIDIA Docs](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html#wwpID0E0R40HA))

**H.264:** (efficient hardware acceleration)
```bash
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, \
  format=(string)NV12, framerate=(fraction)30/1' ! nvv4l2h264enc \
  maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink \
  location=<filename_h264.mp4> -e
  ```

**H.265**: (efficient hardware acceleration)
```bash
gst-launch-1.0 nvarguscamerasrc ! \
  'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, \
  format=(string)NV12, framerate=(fraction)30/1' ! nvv4l2h265enc \
  bitrate=8000000 ! h265parse ! qtmux ! filesink \
  location=<filename_h265.mp4> -e
  ```

### Create virtual loopback stream from NVARGUSCAMSRC

```bash 
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=I420' ! nvvidconv output-buffers=4 ! 'video/x-raw, width=1920, height=1080, framerate=30/1, format=UYVY' ! identity drop-allocation=true ! v4l2sink device=/dev/video4
```

**Start recording**:
```bash
gst-launch-1.0 v4l2src device=/dev/video4 ! 'video/x-raw,width=1920, height=1080, framerate=30/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=I420' ! nvv4l2h264enc maxperf-enable=1 bitrate=8000000 ! h264parse ! qtmux ! filesink location=/OUTPUT.mp4 -e
```

(**End recording safely**):
For the output file to be readable, 'EOS' has to be written at the very end of the file.
When running the above pipeline, a SIGINT (ctrl-c) signal will safely stop the recording and write the EOS byte.
Here is a script to find all recording scripts and stop them:
```bash
pkill -e --signal SIGINT gst-launch-1.0
```

**Create second virtual stream for CSI**:
```bash
gst-launch-1.0 v4l2src device=/dev/video4 ! v4l2sink device=/dev/video5
```

**Run jetson-inference on virtual stream**:
```bash
./detectnet /dev/video5
```

