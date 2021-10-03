# gi-edge
Top level repo for computer vision running on Jetson Nano edge device

## Installation 
See `codename-gimondi/gi-dev-setup` for Jetson Nano setup instructions.

Install YOLOX (see installation instructions from gi-YOLOX)

## Example 

```bash
python3 main.py video -f exps/nano-alpha.py --trt --path ~/maroz-gumuz.mp4
```

## Performance

Performance is measured on test video which is intentionally edited to have lots of objects and passing events.

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

### Real-time goal

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

