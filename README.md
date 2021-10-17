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
## Installation 
See `codename-gimondi/gi-dev-setup` for Jetson Nano setup instructions.

Clone this repository with submodules:
```bash
git clone --recursive git@github.com:codename-gimondi/gi-edge.git
```

[Install YOLOX (see installation instructions from upstream repo)](https://github.com/Megvii-BaseDetection/YOLOX#readme)

## Run Example

Download example / benchmark video from [Google Drive](https://drive.google.com/drive/folders/1e3EbNgNbieoMMaJbaPvsRraDk2AW9iou?usp=sharing)
+ `short-passing.mp4` is 14 seconds long, and used for the below benchmarks
+ `long-passing.mp4` is 2 min 34 seconds long and contains many examples of passing.

```bash
python3 main.py video -f exps/nano-alpha.py --trt --path 2021-10-03-passingcut.mp4
```

## Performance

Performance is measured on '2021-10-03-passingcut.mp4'. This video is edited so that there are many cyclists and passing events.

### Initial pre-optimization
+ all-Python code
+ `nano-alpha.py` model

Step | Time (ms)
--- | ---
Read video from one camera (or video file) | 3 (27)
YOLOX pre-processing | 10
YOLOX model inference | 21
YOLOX post-processing | 14
SORT pre-process | 2
SORT | 13
Watchout | 0.3

Average FPS: 13.5 

### In progress

#### Planned (Hoped) Optimizations

+ NVIDIA DeepStream: (https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/blob/master/HOWTO.md)
+ Paul Bridger optimizations (https://paulbridger.com/posts/video-analytics-deepstream-pipeline/) 
+ Use unified CPU-GPU memory (reduce transfer time): "In order to use Unified Memory you just need to replace malloc() calls with cudaMallocManaged()" (https://www.fastcompression.com/blog/jetson-zero-copy.htm)

#### Completed Optimizations

+ Line 161 in `data_augment.py: preproc()`: Switch cv2 resize interpolation from 'Linear' to 'Nearest' (save 1.5ms)

### Goal

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

