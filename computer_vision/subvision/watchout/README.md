# Watchout

Distance calculation from bounding boxes.

**Input:**
+ Bounding box coordinates
+ Object category expected height/width information
+ Camera field of view information
+ Camera installation height information

**Output:**
+ Distance of each object from camera in meters.


## Implementation

**Ensemble Approach**: Three different kinds of triangulation are used and averaged.

The three methods each use:
+ bounding box height
+ bounding box width
+ bounding box bottom line (only available if the bottom of the object is below the horizon (half height of frame)

