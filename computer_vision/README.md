# Computer Vision

## Directory Structure

```
.
├── debug_utils: timer class, etc.
├── PARAMETERS.py: empirical parameter information (measure things directly)
├── README.md: this file
├── subvision: parts of vision processing pipeline (called from vision.py)
├── vision.py: top-level vision
├── insight: code for visualization, presentation, and development
├── YOLOX: Megvii YOLOX library
├── yolox_exps: neural network definitions for trained YOLOX networks
└── YOLOX_outputs: neural network weights, and saved inference results
```

## Run Video File Example

Download example / benchmark video from [Google Drive](https://drive.google.com/drive/folders/1e3EbNgNbieoMMaJbaPvsRraDk2AW9iou?usp=sharing)
+ `short-passing.mp4` is 14 seconds long, and used for the below benchmarks
+ `long-passing.mp4` is 2 min 34 seconds long and contains many examples of passing.
+ `long-being-overtaken.mp4` is 3 min 52 seconds and contains many examples of being passed, and also some more complex scenes where passing doesn't actually occur.


## From Training to Inference

1. (desktop) Define parameters to train a new neural network by creating a new python file at `YOLOX/exps/`.
2. (desktop) Train the neural network with `YOLOX/tools/train.py`.
3. (desktop) The results of training are saved to `YOLOX/YOLOX_outputs/`.
4. (jetson) Copy python files from desktop `YOLOX/exps` to `gi-edge/computer_vision/hydo_exps`.
5. (jetson) Copy training results (weights) from desktop `YOLOX/YOLOX_outputs` to `gi-edge/computer_vision/YOLOX/YOLOX_outputs`.

Run inference

```bash
python3 vision.py -vid0 ~/Videos/long-passing.mp4 -vid1 ~/Videos/long-being-overtaken.mp4 -f yolox_exps/nano-alpha.py --view_result --save_result
```
+ the `-f` argument will retrieve neural network architecture information from `exps` file
+ and also automatically find the best checkpoint from `YOLOX_outputs`

In order to use TensorRT (device-specific NVIDIA optimizations), use `tools/trt.py` in YOLOX_tt repository. If you get a `'NoneType' object has no attribute 'serialize` error, it's because the workspace size is too big and memory ran out.
