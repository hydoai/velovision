import numpy as np
import math
from configparser import ConfigParser
import os
'''
Send warning if size of bounding box gets bigger at a certain rate.

Initialize new object for each camera stream,
then call 'step()' for every detection.

Relies on modified SORT, where the derivatives and object classes are made public.
'''

class Watchout:

    def __init__(self, names, names_to_height_dict, width, height, buffer_len=10, area_min_delta=40, area_max_delta=4000, left_hand_drive=False):
        #set attributes
        self.names = names # of detect object categories
        self.names_to_height_dict = names_to_height_dict
        self.width = width
        self.height = height
        self.buffer_len = buffer_len
        self.area_min_delta = area_min_delta
        self.area_max_delta = area_max_delta
        self.left_hand_drive = left_hand_drive
        
        #rolling buffer
        self.rolling_filter = [False for _ in range(buffer_len)] #queue filter to reduce false positives.
        
        
    def step(self, tracked_dets=np.empty((0,9))):
        '''
        Parameters:
        tracked_dets, an np.array where
            index: information
            0 : x1
            1 : y1
            2 : x2
            3 : y2
            4 : category
            5 : d(x_center)/dt
            6 : d(y_center)/dt
            7 : d(area)/dt
            8 : object_id (tracked id)
        
        Returns
            same as tracked det, but appended with '9:distance' (CURRENTLY UNIMPLEMENTED)

        
        '''
        all_threats = np.empty((0,10))
        warned_threats = {} #record how many times each id was honked at. Avoids too repetitive honking.
        

        for obj in tracked_dets:  #for every tracked object

            # HEIGHT BASED DISTANCE; units are consistent and realistic
            # NOT IMPLEMENTED!

            # estimate distance from height and hard-coded height for each object category
            if self.names[int(obj[4])] in self.names_to_height_dict.keys():
                hard_coded_obj_height = self.names_to_height_dict[self.names[int(obj[4])]]
            image_obj_height_pix = (obj[3]-obj[1])
            #FOV_height_rad = math.radians(float(self.camera_config.get('FRONT CAM','FOV_height_deg')))
            #distance = (self.front_cam_frame_height * obj_height_est)/(obj_height_pix * math.tan(FOV_height_rad)).item()
            distance = 123456789 # placeholder


            # AREA BASED PROXIMITY; units are meaningless
            delta_bbox_area = obj[7].item()
            
            x_center = (obj[0]+((obj[2]-obj[0])/2)).item()
            y_center = (obj[1]+((obj[3]-obj[1])/2)).item()

            # filter out any bboxes whose center isn't in region of interest
            if not self.left_hand_drive: # right hand drive: pass on the left
                left_limit = self.width*(4/9)
                #right_limit = self.width*(6.5/9)
                right_limit = self.width*(9/9)
            else:
                left_limit = self.width*(2.5/9)
                right_limit = self.width*(5/9)

            if (x_center >= left_limit and x_center <= right_limit):
                append_distance = np.hstack((obj, np.array((distance))))
                all_threats = np.vstack((all_threats, append_distance))

        if len(all_threats):
            # test bounding box area change against threshold
            any_exceed_lower_thresh = np.max(all_threats[:,7]).item() > self.area_min_delta
            any_under_upper_thresh = np.max(all_threats[:,7]).item() < self.area_max_delta

            if any_exceed_lower_thresh and any_under_upper_thresh:

                #warn_cats.append(all_threats[np.argmax(all_threats[:,2]), 1].tolist()) #list threatening object category numbers.
                # 'true' into filter queue
                updated_filter = self.rolling_filter[1:]
                updated_filter.append(True)
                self.rolling_filter = updated_filter
            else:
                # 'false' into filter queue
                updated_filter = self.rolling_filter[1:]
                updated_filter.append(False)
                self.rolling_filter= updated_filter

        # once buffer is full, send alert
        if all(self.rolling_filter):
            return all_threats
        else:
            return None

