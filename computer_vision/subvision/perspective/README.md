# Perspective: Predicting overtaking encounter from top view perspective

## Plan View

Convert (distance, angle from center) to a top-down view of the situation (x,y)

Visual explanation:

**Input:** 
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

**Output:**
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

Then, the (x,y) coordinate values for each object are fed into a Kalman Filter, which returns (x,y,dx/dt,dy/dt) for the next time step.

This value is is used to calculate two overtaking-relevant features:

+ **time_to_encounter** = x / (dx/dt)
+ **passing_separation (y-intercept)** = (dy/dt) * time_to_encounter



