# gi-edge
Top level repo for computer vision running on Jetson Nano edge device

## Repository Schematic
Created with [asciiflow](https://asciiflow.com)
```
┌───────────────────┐      ┌──────────────┐
│Jetson Installation│      │Mock Resources│
└───────────────────┘      └──────────────┘

┌─────────────────────────────────────────────────────┐
│Live Session Code                                    │
│ ┌─────────────────┐   ┌──────────────────────────┐  │
│ │Sensing Interface│   │ User Interface (UI) Input│  │
│ └───────────────┬─┘   └───┬──────────────────────┘  │
│                 │         │                         │
│               ┌─▼─────────▼───┐                     │
│               │Computer Vision│                     │
│               └─┬─────────┬───┘                     │
│                 │         │                         │
│ ┌───────────────▼──┐   ┌──▼───────────────────────┐ │
│ │Feedback Interface│   │User Interface (UI) Output│ │
│ └──────────────────┘   └──────────────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘

┌────────────────┐
│Blackbox Access │
└────────────────┘
```
+ [Jetson Installation](jetson_installation): Installing required packages
+ [Mock Resources](mock_resources): Mock videos, etc. for desktop development and testing

**Live Session Code**: Code that runs when Hydo is on and active.

+ [Sensing Interface](sensing_interface): Camera info, gstreamer pipeline setup
+ [UI Input](ui_input): Buttons and switches for user input
+ [Computer Vision](computer_vision): Vision neural network + postprocessing 
+ [Feedback Interface](feedback_interface): Speaker (warning sounds) interfacing
+ [UI Output](ui_output): Display output interfacing

+ [Blackbox Access](blackbox_access): When you plug in the Jetson to a computer, it registers as a USB drive containing the recorded videos.

## Todo / Hopeful Improvements

+ NVIDIA DeepStream: (https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md)
+ Paul Bridger optimizations (https://paulbridger.com/posts/video-analytics-deepstream-pipeline/) 
+ Use unified CPU-GPU memory (reduce transfer time): "In order to use Unified Memory you just need to replace malloc() calls with cudaMallocManaged()" (https://www.fastcompression.com/blog/jetson-zero-copy.htm)

## Performance Goal

Step | Time (ms)
--- | ---
Read video from one camera (or video file) | 3 (27)
YOLOX pre-processing | 3 (3x faster)
YOLOX model inference | 21
YOLOX post-processing | 5 (3x faster)
SORT pre-process | 2
SORT | 4 (3x faster)
Watchout | 0.3

Average FPS: 25

