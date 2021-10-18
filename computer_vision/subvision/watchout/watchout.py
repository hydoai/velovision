import numpy as np
import math
import os

class Watchout:
    '''
    Send warning sound to user if area of bounding box is increasing at a defined rate
    
    Usage:
        sort = Sort()
        watchout = Watchout()
        
        for frame in video:
            detections = yolo_detector(frame)
            tracked_dets = sort(detections)
            sound_alarm = watchout(tracked_dets) # a boolean whether to sound alarm right now
    
    '''

    def __init__(
            self,
            index_to_name_dict,
            cam0_width,
            cam0_height,
            cam1_width,
            cam1_height,
            buffer_len = 10,
            area_min_delta = 40
            area_max_delta = 4000
            left_hand_drive = False
            ):
        self.index_to_name_dict = index_to_name_dict
        self.cam0_width = cam0_width
        self.cam0_height = cam0_height
        self.cam1_width = cam1_width
        self.cam1_height = cam1_height
        self.buffer_len = buffer_len
        self.area_min_delta = area_min_delta
        self.area_max_delta = area_max_delta
        self.left_hand_drive = left_hand_drive

        self.rolling_filter = [False for _ in range(buffer_len)] # queue filter to reduce false positives

    def step(self, tracked_dets = np.empty((0,9))):
        '''
        Parameters


