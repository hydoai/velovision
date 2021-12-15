TODO

- [ ] Run on Jetson to validate camera input
- [ ] Cover timing tightly on inference loop, optimize (for example, don't needlessly render visualizations)
- [ ] Write unit tests for subvision parts

# gi-edge
Computer vision-based overtake warning system for bicycles.

## Notes on Coding Style

In many cases, simple examples of module functionality is self-documented. So if you want to understand what a module does in isolation, just run that module from your terminal and read the code following `if __name__ == "__main__":`.

## Quickstart on Desktop

**Clone this repository**

```bash
git clone git@github.com:codename-gimondi/gi-edge.git
```

**Create conda environment** ([How to install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html))

```bash
conda env remove hydo-dev # in case there is a previous environment named hydo-dev
conda env create -f environment.yaml
```

This creates a conda environment called 'hydo-dev'. Activate it:

```bash
conda activate hydo-dev
```

**Install YOLOX**

```bash
cd gi-edge/temp

git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```

**Download example test videos**

GoPro (1920x1080, 30fps)

+ [Click to download forward-looking bicycle path footage](https://storage.hydo.ai/gi-edge-assets/example-footage/long-overtaking.mp4)
+ [Click to download rearward-looking bicycle path footage](https://storage.hydo.ai/gi-edge-assets/example-footage/long-being-overtaken.mp4)

Or, run on the command line:

```bash
wget -O ~/Downloads https://storage.hydo.ai/gi-edge-assets/example-footage/long-overtaking.mp4
wget -O ~/Downloads https://storage.hydo.ai/gi-edge-assets/example-footage/long-being-overtaken.mp4
```

Devkit-One Cameras (front: 1280x720 30fps, rear: 640x480 30fps)

+ [Download front](https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/front-2021-12-04.mkv)
+ [Download rear](https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/rear-2021-12-04.mkv)

Or, run command:

```bash
wget -O ~/Downloads https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/front-2021-12-04.mkv
wget -O ~/Downloads https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/rear-2021-12-04.mkv
```

**View real-time inference output**

```
cd gi-edge/computer_vision
python3 vision.py -vid0 ~/Downloads/long-overtaking -vid1 ~/Downloads/long-being-overtaken.mp4 -f yolox_exps/nano-foxtrot.py --view_result
```

### Run automated tests

Ensure that you are in the `computer_vision` directory, and that you have `hydo-dev` conda environment activated. Then run:

```
python3 -m pytest
```
You will probably hear some sounds (front and rear warning sounds) while the test is running.

## Getting Started on NVIDIA Jetson

Please see [hydoai/dk-1setup](https://github.com/hydoai/dk1-setup) for instructions on running on Jetson.

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
```


### Live Session Code: Code that runs when Hydo is on and active.

+ [Sensing Interface](sensing_interface): Set up NVIDIA hardware accelerated camera video input stream. Implement always-on hardware accelerated video saving.
+ [UI Input](ui_input): Interface with buttons and switches on the case.
+ [Computer Vision](computer_vision): Object detection neural network, tracking, and warning algorithms. 
+ [Feedback Interface](feedback_interface): Interface with warning sound speaker.
+ [UI Output](ui_output): Interface with OLED display

### Other Non-Live Code

+ [Jetson Installation](jetson_installation): Install required packages for running the live session code.
+ [Mock Resources](mock_resources): Mock test video files, etc. for desktop development and testing

## Videos


[Project Gimondi Work in Progress 02](https://youtu.be/eCJcu_2iLPg)

[Project Gimondi Work in Progress 01](https://youtu.be/SEfXO2w2qVI)

[Hydo first try](https://youtu.be/Jk-cQkcG4iY)
