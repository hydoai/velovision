
![banner](https://github.com/hydoai/brand-id/raw/main/velovision/velovision-banner-pictures.png)

---

<!-- Run ./update-readme-toc.sh -->
<!-- or: doctoc --title '# Table of Contents' --maxlevel 1 README.md -->
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
# Table of Contents

- [Running on a bike](#running-on-a-bike)
- [Running on a PC](#running-on-a-pc)
- [Testing](#testing)
- [Development Tips](#development-tips)
- [Licensing](#licensing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

---

## What is HYDO velovision?

HYDO velovision is the world's first computer vision-based active cycling assistance system. It can easily be attached to any bicycle to provide Automatic Overtake Warning sound for  pedestrians, cyclists, and vehicles. It's an ADAS (advanced driver assistance system) for bicycles.

HYDO velovision is created by [Jason Sohn](https://jasonsohn.com).

## HYDO in Action

![](readme_assets/foxtrot-intercept-vis.gif)

<table>
  <tr>
    <td><a href="https://youtu.be/ND56-PTQYCA" title="chestcam-1"><img src="readme_assets/chestcam-thumbnail-1.png"></a></td>
    <td><a href="https://youtu.be/GUyWp-IDISc" title="chestcam-2"><img src="readme_assets/chestcam-thumbnail-2.png"></a></td>
    <td><a href="https://youtu.be/BhYqEL78wPo" title="chestcam-3"><img src="readme_assets/chestcam-thumbnail-3.png"></a></td>
    <td><a href="https://youtu.be/MSXN9TsbpYU" title="chestcam-4"><img src="readme_assets/chestcam-thumbnail-4.png"></a></td>
  </tr>
</table>

## Project Milestones

+ 2020-11: Initial prototype is created.
+ 2021-09: [Jason Sohn](https://jasonsohn.com) receives KRW 7,000,000 in funding for this project from the South Korean government via K-startup grant.
+ 2021-11-25: Patent submitted: "A BICYCLE WARNING DEVICE THROUGH OBJECT AUTOMATIC RECOGNITION AND A METHOD OF OVERTAKING WARNING USING THE SAME". Patent number: 10-2021-0163981.
+ 2021-12: This project graduates the K-startup program with honors.
+ 2021-12: **velovision** is made open source.
+ 2022-01: Development on this project is paused as Jason focuses on neuroscience research. [See latest on his blog](https://jasonsohn.com)

## More facts about velovision

+ **velovision** is mostly written in Python.
+ PyTorch is my deep learning framework of choice.
+ The neural network is trained on a synthetic dataset derived from over 300GB of self-collected cycling footage.
+ This custom dataset is larger than COCO 2017 train dataset.
+ The deployed neural network is highly optimized with NVIDIA TensorRT.
+ HYDO Devkit-One, the first hardware platform for velovision, uses a Jetson Xavier NX processor and uses less than 15 watts of power.
+ The [hydo.ai](https://hydo.ai) website is also [available in Korean](https://kr.hydo.ai)

![](readme_assets/dataset-wall.png)

*A slice of the large video dataset, before autolabeling*

# Running on a bike
![velovision deploy tests](https://github.com/hydoai/velovision/actions/workflows/velovision_deploy_tests.yml/badge.svg)

[HYDO Devkit-One](https://hydo.ai) is pre-installed with velovision pre-installed and ready to ride.



<details>
  <summary> Developers only
  </summary>
  
See [hydoai/dk1-setup](https://github.com/hydoai/dk1-setup) for ground-up setup of Devkit-One.

</details>

# Running on a PC (development)

![velovision develop tests](https://github.com/hydoai/velovision/actions/workflows/velovision_develop_tests.yml/badge.svg)

**Clone this repository**

```bash
git clone git@github.com:hydoai/velovision.git
```

**Create conda environment** ([How to install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html))

```bash
conda env remove hydo-dev # in case there is a previous environment named hydo-dev
conda env create -f environment.yaml
conda activate hydo-dev
```

**Install YOLOX**

```bash
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

mkdir temp
cd temp

git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```

**Download example test videos**

Two sets of videos are used as examples. `Right click -> Save Link As...` to download.
1. GoPro (1920x1080, 30fps)
2. Devkit-One Cameras (front: 1280x720 30fps, rear: 640x480 30fps)

(links are currently broken; fix coming soon)
+ [GoPro front](https://storage.hydo.ai/gi-edge-assets/example-footage/long-overtaking.mp4)
+ [GoPro rear](https://storage.hydo.ai/gi-edge-assets/example-footage/long-being-overtaken.mp4)
+ [DevKit-One front](https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/front-2021-12-04.mkv)
+ [DevKit-One rear](https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/rear-2021-12-04.mkv)

<details>
  <summary> Alternatively, download via `wget`
  </summary>

```bash
wget -O ~/Downloads/long-overtaking.mp4 https://storage.hydo.ai/gi-edge-assets/example-footage/long-overtaking.mp4
wget -O ~/Downloads/long-being-overtaken.mp4 https://storage.hydo.ai/gi-edge-assets/example-footage/long-being-overtaken.mp4

wget -O ~/Downloads/front-2021-12-04.mkv https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/front-2021-12-04.mkv
wget -O ~/Downloads/rear-2021-12-04.mkv https://storage.hydo.ai/gi-edge-assets/first-blackbox-recordings/rear-2021-12-04.mkv
```
  
</details>

**Run main script**

```
cd velovision/computer_vision
python3 vision.py -vid0 ~/Downloads/long-overtaking -vid1 ~/Downloads/long-being-overtaken.mp4 -f yolox_exps/nano-foxtrot.py --view_result
```

# Testing

## Run manual tests

Ensure that you are in the `computer_vision` directory, and that you have `hydo-dev` conda environment activated. Then run:

```
python3 -m pytest
```
You might hear some sounds (front and rear warning sounds) while the test is running.

## Run automated tests

See [docs/TESTING.md](docs/TESTING.md) to learn more.

To run the same suite of automated tests on your local machine,

[Install nektos/act](https://github.com/nektos/act), then run

```
cd computer_vision
act
```

# Development Tips

## Todo List

- [ ] Set up Jetson with identical hardware for automated tests (github runner)
- [ ] Write unit tests for subvision parts

## Repository Structure

```
┌─────────────────────────────────────────────────────┐
│ Live Inference Code                                 │
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
*diagram using [asciiflow](https://asciiflow.com)*

+ [Sensing Interface](sensing_interface): Set up camera inputs. Hardware accelerated encoding full FOV front and rear cameras.
+ [UI Input](ui_input): Interface with buttons and switches on the case.
+ [Computer Vision](computer_vision): Object detection neural network, tracking, and warning algorithms. 
+ [Feedback Interface](feedback_interface): Interface with warning sound speaker.
+ [UI Output](ui_output): Interface with OLED display

## Getting started tips

+ The main script is `computer_vision/vision.py`. All submodules are called from there.
+ In many cases, modules are self-testable. So if you want to understand what a module like `from subvision.intercept import intercept` does in isolation, just run that module from your terminal like `python3 intercept.py` and read the code following `if __name__ == "__main__":`. This should give you a good idea of what it does, and also be a 'playground' for you to make temporary experiments.
+ Installing on desktop simply install YOLOX, then installing `hydo-dev` conda environment. However, installing on NVIDIA Jetson platform is more involved and takes much longer. Many usually pip installable packages must be installed from apt and requires other dependencies.


# Licensing

velovision is released under the Apache-2.0 open source license. Some parts of the software are released under other licenses as specified; all of the code is permissively licensed.
