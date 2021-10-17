# Arducam 1080P Low Light WDR USB Camera Module for Computer, 2MP 1/2.8” CMOS IMX291 100 Degree Wide Angle Mini UVC Webcam Board with Microphone

![arducam_product_image](https://www.arducam.com/wp-content/uploads/2019/11/b0200-1.jpg)

Official website: [link](https://www.arducam.com/product/arducam-1080p-low-light-wdr-usb-camera-module-for-computer-2mp-1-2-8-cmos-imx291-100-degree-wide-angle-mini-uvc-spy-webcam-board-with-microphone-3-3ft-1m-cable-for-windows-linux-mac-os/)

## Minimal Use

Show list of connected devices (cameras)
```bash
ls /dev/video*
```

Figure out what formats are supported for a given camera (example uses /dev/video1)
```bash
sudo apt install v4l-utils
v4l2-ctl -d /dev/video1 --list-formats-ext
```
See camera stream in a window
```bash
gst-launch-1.0 device=/dev/video1 ! xvimagesink
```

## Real Life Use

+ Add NVIDIA hardware accelerated MJPEG decoding


### NVIDIA-only pipeline

Read image frames into GPU memory and display it from there. The stream is displayed as full screen window.
```bash
gst-launch-1.0 v4l2src device=/dev/video1 io-mode=2 ! image/jpeg, width=1280, height=720, framerate=30/1, format=MJPG ! nvjpegdec ! video/x-raw ! nvvidconv ! 'video/x-raw(memory:NVMM), width=1280, height=720' ! nvoverlaysink
```

### Hardware Accelerated, but Universal output pipeline

Similar to above, but converts the stream into a more common, universal format.
```bash
gst-launch-1.0 v4l2src device=/dev/video1 io-mode=2 ! image/jpeg, width=1280, height=720, framerate=30/1, format=MJPG ! nvjpegdec ! video/x-raw ! videoconvert ! xvimagesink
```

Open up `jtop` to see that under [HW engines], NVJPG is on.
```bash
sudo -H pip install -U jetson-stats # installation
jtop
```
