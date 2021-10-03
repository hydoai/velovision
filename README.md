# gi-edge
Top level repo for computer vision running on Jetson Nano edge device

## Installation 
See `codename-gimondi/gi-dev-setup` for Jetson Nano setup instructions.

Install YOLOX (see installation instructions from gi-YOLOX)

## Run Example

Download example / benchmark video from [Google Drive: '2021-10-03-passingcut.mp4'](https://drive.google.com/file/d/1Z5rp_d7r6JPkoP2mBSLsu9PbdELaxwt2/view?usp=sharing)

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

#### Optimizations

+ Line 161 in `data_augment.py: preproc()`: Switch cv2 resize interpolation from 'Linear' to 'Nearest' (save 1.5ms)

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

