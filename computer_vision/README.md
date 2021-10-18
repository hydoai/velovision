#Computer Vision

Core image processing and actionable output producing code.

## Install YOLOX

```bash
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```

Install pycocotools
```bash
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```


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
