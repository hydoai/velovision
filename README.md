# gi-edge
Top level repo for all Hydo functionality on Jetson Xavier NX device operating on the edge (on bike).

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


