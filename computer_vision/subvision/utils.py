from enum import Enum
import numpy as np
import cv2

class CameraFacing(Enum):
    FRONT = 0
    REAR = 1

class FrameHalf(Enum):
    LEFT = 0
    RIGHT = 1

def center_crop(image, width, height, nudge_down=0, nudge_right=0):
    x = image.shape[1]/2 - width/2
    y = image.shape[0]/2 - height/2
    return image[int(y+nudge_down):int(y+height+nudge_down) , int(x+nudge_right):int(x+width+nudge_right)]

def fixed_aspect_ratio_crop(frame, camera_info, to_width, to_height):
                '''
                Crop and resize frame to width * height, zooming in by crop_ratio
                If the aspect ratio of frame does not match to_width/to_height,
                then we crop based on width only.
                '''
                crop_width = int(frame.shape[1] / camera_info.inference_crop_ratio)
                crop_height = int(crop_width * (to_height/to_width))

                cropped_frame = center_crop(image=frame,
                                            width=crop_width,
                                            height=crop_height,
                                            nudge_down = camera_info.nudge_down,
                                            nudge_right=camera_info.nudge_right)
                resized_frame = cv2.resize(
                    cropped_frame,
                    (to_width, to_height),
                    interpolation=cv2.INTER_LINEAR)
                return resized_frame

def combine_dets(front_dets, rear_dets, height):
    '''
    Input:
        The shape of front_dets and rear_dets is (num_detections, 9)
        for each row, the index corresponds to
            0 : x1 (within raw_img shape)
            1 : y1 (within raw_img shape)
            2 : x2 (within raw_img shape)
            3 : y2 (within raw_img shape)
            4 : class index
            5 : x_center # added
            6 : y_center # added
            7 : assigned to front (0) or rear(1)
            8 : track id
            9 : distance

    Output:
        hstacked dets, where the rear_dets coordinates are shifted down
    '''
    rear_dets[:,(1,3,6)] = rear_dets[:,(1,3,6)] + height
    return np.vstack((front_dets, rear_dets))

def split_dets(dets, width, height):
    '''
    Summary: (extend screen width to see correctly)
    ┌─────────────────────────────────┐
    │                                 │
    │                                 │
    │    ┌────────┐                   │
    │    │        │                   │       ┌────────────────────────────────┐     ┌────────────────────┬────────┬──┐
    │    │        │                   │       │                                │     │                    │        │  │
    │    │        │                   │       │                                │     │                    │  rear  │  │
    │    │        │                   │       │    ┌─────────┐                 │     │     ┌─────────┐    │        │  │
    │    │        │     ┌────────┐    │       │    │         │                 │     │     │         │    │        │  │
    │    │        │     │        │    │       │    │         │                 │     │     │  rear   │    └────────┘  │
    │ ── └────────┴─ ── │        ├── ─┤  ──►  │    │         │                 │     │     │         │                │
    │                   │        │    │       │    │  front  │                 │     │     └─────────┘                │
    │                   │        │    │       │    │         │                 │     │                                │
    │     ┌─────────┐   │        │    │       │    │         │                 │     │                                │
    │     │         │   │        │    │       └────┴─────────┴─────────────────┘     └────────────────────────────────┘
    │     │         │   └────────┘    │
    │     │         │                 │
    │     └─────────┘                 │
    │                                 │
    │                                 │
    └─────────────────────────────────┘
    Input:
        dets are vertically stacked (front video on top, rear video at the bottom)
        width and height of video (assumed to be the same for front and rear)


        The shape of dets is (num_detections, 9)
        for each row, the index corresponds to
            0 : x1
            1 : y1
            2 : x2
            3 : y2
            4 : class index
            5 : empty (0)
            6 : empty (0)
            7 : empty (0)
            8 : track id

    Output:
        (front dets, rear dets), where each contain boxes within (0,width), (0,height) range.

    In addition, any bounding boxes that go beyond their origin video frames are cut so that they don't stretch beyond it.

        The shape of outputs is (num_detections, 9)
        for each row, the index corresponds to
            0 : x1 (within raw_img shape)
            1 : y1 (within raw_img shape)
            2 : x2 (within raw_img shape)
            3 : y2 (within raw_img shape)
            4 : class index
            5 : x_center # added
            6 : y_center # added
            7 : assigned to front (0) or rear(1)
            8 : track id

    '''

    # development
    dets = np.round(dets)
    input_dets = dets

    # calculate x and y centers
    dets[:,5] = (dets[:,2] + dets[:,0])/2
    dets[:,6] = (dets[:,3] + dets[:,1])/2

    # assign box to front(0) or rear(1)-facing camera
    dets[:,7] = dets[:,6] > (height)

    front_dets = dets[np.where(dets[:,7] == 0)]
    rear_dets = dets[np.where(dets[:,7] == 1)]

    # move rear coordinates back (un-stack)
    rear_dets[:,(1,3,6)] -= height

    # extra: trim any spillover bounding boxes bits
    for dets in (front_dets, rear_dets):
        dets[:, 3:4] = np.where(dets[:,3:4] > height, height, dets[:,3:4]) # trim any below bottom limit
        dets[:, 3:4] = np.where(dets[:,3:4] < 0, 0, dets[:,3:4]) # trim any above top limit

    return front_dets, rear_dets

