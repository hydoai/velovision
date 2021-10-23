# Intercept

## Kalman Filter Object Trajectory Prediction

Then, the (x,y) coordinate values for each object are fed into a Kalman Filter, which returns (x,y,dx/dt,dy/dt) for the next time step.

## Overtaking-relevant feature calculation

This value is is used to calculate two overtaking-relevant features:

+ **time_to_encounter** = x / (dx/dt). "How many seconds in the future will the pass occur?"
+ **passing_separation (y-intercept)** = (dy/dt) * time_to_encounter. "How close will the pass be?"


