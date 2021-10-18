# Computer Vision

Core image processing and actionable output producing code.

## Development Right Now

Running vision script on test videos.
```bash
python3 vision.py --vid0 ~/Videos/long-passing.mp4 --vid1 ~/Videos/long-being-overtaken.mp4 -f YOLOX/exps/hydo/nano-alpha.py --save_result
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
