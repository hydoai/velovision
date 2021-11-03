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
