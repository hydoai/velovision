# Insight

Code for displaying the internal workings of this project.

Intended for development and presentation.
=======
# Insight: Visualizing the internal workings of HYDO Computer Vision for presentations and development.

## Perspective View to Plan View Conversion

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

## Kalman Filter Object Trajectory Prediction

Then, the (x,y) coordinate values for each object are fed into a Kalman Filter, which returns (x,y,dx/dt,dy/dt) for the next time step.

## Overtaking-relevant feature calculation

This value is is used to calculate two overtaking-relevant features:

+ **time_to_encounter** = x / (dx/dt)
+ **passing_separation (y-intercept)** = (dy/dt) * time_to_encounter

This is the final interface. Modify these two values to change program behavior.
>>>>>>> b587002fe8b4f0891db700cb94c213d804c6a980
