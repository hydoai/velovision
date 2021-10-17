
## Run Mock Video Example

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

