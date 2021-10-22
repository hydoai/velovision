# Insight: Visualizing the internal workings of HYDO Computer Vision for presentations and development.

## Plan View

Convert (distance, angle from center) to a top-down view of the situation (x,y)

Visual explanation:

Input: 
```
┌─────────Front_Camera_Frame──────────┐
│                                     │
│                                     │
│                                     │
│                                     │
│       ┌─────────┐                   │
│       │         │                   │
│       │   obj   │                   │
│       │         │..                 │
│       │(  5m,   │.....              │
│       │ 10 deg )│.........          │
│       └─────────┘............       │
│      ...........................    │
│   ................................  │
│.....................................│
└─────────────────────────────────────┘
```
+ Distance from the camera calculated using Triangulated Ensemble (previous step)
+ Degrees calculated by: cam_angle[rad] * (( (frame_width/2) - (obj_center_x)) / (frame_width))

```
                 Left
       @@@         ▲         ***
       @@@@@       y       *****
       @@@@@@@     │     ***┌───┐
       @@@@@@@@@   │   *****│obj│
       @@@@@@@@@@@ │ *******└───┘
Rear   @@@@@@@@@@┌─┴─┐********** Front
─────────────────┤ego├─────────────x►
       @@@@@@@@@@└─┬─┘**********
       @@@@@@@@@@@ │ ***********
       @@@@@@@@@   │   *********
       @@@@@@@     │     *******
       @@@@@       │       *****
       @@@         │         ***
                  Right

 * Front Field of View
 @ Rear Field of View

```


