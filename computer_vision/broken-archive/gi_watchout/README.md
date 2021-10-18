# Watchout

Selective warning thresholding from abewley/SORT kalman filter information by [Jason Sohn (neuroquantifier)](https://github.com/neuroquantifier)

## Example use of Watchout in pseudocode (not verified; TODO check real life)

```python3
from watchout import *

names = {'car', 'person'} # dictionary of categories
names_to_height_dict = {'car':2, 'person': 1.7}
width = 1280
height = 720


watchout = Watchout(names, names_to_height_dict, width, height)

for frame in video:
    detections = yolox(frame) # object detection
    sort_output = sort(detections) # SORT: Simple Online Realtime Tracker
    
    # sort_output should be a 2d numpy array, where each row is:
    # [x1, y1, x2, y2, category, d(x_center)/dt, d(y_center)/dt, d(area)/dt, object_id]
    watchout_output = watchout.update(sort_output)
    # watchout_output returns the same kind of array
    
```
