# gi-edge
Computer vision-based overtake warning system for bicycles.

## Quickstart on Desktop

**Clone this repository**

```bash
git clone --recurse-submodules git@github.com:codename-gimondi/gi-edge.git
```

**Download example test videos**

+ [Click to download forward-looking bicycle path footage](https://storage.hydo.ai/gi-edge-assets/example-footage/long-overtaking.mp4)
+ [Click to download rearward-looking bicycle path footage](https://storage.hydo.ai/gi-edge-assets/example-footage/long-being-overtaken.mp4)

Or, run on the command line:

```bash
wget -O ~/Downloads https://storage.hydo.ai/gi-edge-assets/example-footage/long-overtaking.mp4
wget -O ~/Downloads https://storage.hydo.ai/gi-edge-assets/example-footage/long-being-overtaken.mp4
```

**View real-time inference output**

```
cd gi-edge/computer_vision
python3 vision.py 
```

## Getting Started on NVIDIA Jetson

Get the Jetson ready for deep learning inference in general by following [hydoai/dk-1setup](https://github.com/hydoai/dk1-setup).

While the repositories are private, execute `generate-github-ssh-key.sh`
```bash
chmod 777 generate-github-ssh-key.sh
./generate-github-ssh-key.sh
```

Install [gi-YOLOX](https://github.com/codename-gimondi/gi-YOLOX)
```bash
git clone git@github.com:codename-gimondi/gi-YOLOX.git
cd gi-YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

```

Basically, this allows import of `yolox` as a user package.

Download the demo front and rear videos:
```bash
wget -O ~/Videos/long-overtaking.mp4 https://storage.hydo.ai/gi-edge-assets/example-footage/long-overtaking.mp4
wget -O ~/Videos/long-being-overtaken.mp4 https://storage.hydo.ai/gi-edge-assets/example-footage/long-being-overtaken.mp4
```

Clone this repository with submodules

```bash
git clone --recurse-submodules git@github.com:codename-gimondi/gi-edge.git
```

Run example
```bash
cd gi-edge/computer_vision
python3 vision.py -vid0 ~/Videos/long-overtaking.mp4 -vid1 ~/Videos/long-being-overtaken.mp4 -f yolox_exps/nx-foxtrot.py --trt
```

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
