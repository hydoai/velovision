import numpy as np
import math
import os

import sys
sys.path.append("../..")
from PARAMETERS import KNOWN_CLASS_SIZES, CAM_LENS_INFO
from subvision.utils import CameraFacing

class Watchout:
    '''
    Calculate the distance to an object from bounding box coordinates, camera information, and expected object sizes in meters.

    Implements a 'ensemble' approach, where 2~3 different ways of triangulating the distance are averaged out.

    Usage:
        sort = Sort()
        watchout = Watchout()

        for frame in video:
            detections = yolo_detector(frame)
            tracked_dets = sort(detections)
            watchout_results = watchout(tracked_dets) # a boolean whether to sound alarm right now

    '''

    def __init__(
            self,
            index_to_name_dict = {},
            cam0_width = 960,
            cam0_height = 540,
            cam1_width = 960,
            cam1_height = 540,
            buffer_len = 10,
            area_min_delta = 40,
            area_max_delta = 4000,
            left_hand_drive = False,
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

        self.memory = {} # dictionary where key is track_id, and value is a 'Thing' object which contains all the other information.

    def step(self, tracked_dets = np.empty((0,9))):
        '''
        Parameters:
        tracked_dets, a np.array where
            0 : x1
            1 : y1
            2 : x2
            3 : y2
            4 : category index
            5 : d(x_center)/dt
            6 : d(y_center)/dt
            7 : d(area)/dt
            8 : object_id (tracked, unique id)

        Returns:
            tracked_dets +
            9 : distance

        '''
        row, col = tracked_dets.shape
        watchout_output_dets = np.empty((0,col+1))

        for i in range(len(tracked_dets)):
            obj = tracked_dets[i]
            if obj[8].item() in self.memory.keys(): #we've seen this guy before
                watchedthing = self.memory[obj[8].item()]
                watchedthing.step(obj)
                #print(f"Predicted distance: {watchedthing.pred_distance}m")
                obj = np.hstack((obj, watchedthing.pred_distance))

            else:
                self.memory.update({obj[8].item(): WatchedThing(obj)}) # create new Thing

                # if memory contains more than 100, delete 25 earliest entries
                if len(self.memory) > 50:
                    earliest_indexes = [x for x in sorted(self.memory.keys())][:25]
                    for ind in earliest_indexes:
                        del self.memory[ind]
                obj = np.hstack((obj, 100))
            watchout_output_dets = np.vstack((obj, watchout_output_dets))
        return watchout_output_dets


class WatchedThing:
    def __init__(self, obj):
        self.update_basic_properties(obj)

        # history dependent properties
        self.dddt = None # distance change

    def step(self, obj):
        self.update_basic_properties(obj)

    def update_basic_properties(self, obj):
        # verbatim properties
        self.x1 = obj[0].item()
        self.y1 = obj[1].item()
        self.x2 = obj[2].item()
        self.y2 = obj[3].item()
        self.cat_ind = obj[4].item()
        self.dxdt = obj[5].item()
        self.dydt = obj[6].item()
        self.dadt = obj[7].item()

        # calculated properties
        self.area = ((obj[2]-obj[0]) * (obj[3]-obj[1])).item()
        self.x_c = (obj[0] + ((obj[2]-obj[0])/2)).item()
        self.y_c = (obj[1] + ((obj[3]-obj[1])/2)).item()

        self.pred_distance = self.predict_distance_from_box()

    def predict_distance_from_box(self):

        '''

        distance predicted using vertical = known_height(meters) / {tan(camera_angle_vertical(rad) * (object_height(pixels) / frame_height(pixels)) }
        distance predicted using horizontal = known_width(meters) / {tan(camera_angle_horizontal(rad) * (object_width(pixels) / frame_width(pixels)) }
        distance predicted using bottom-center of box = camera_installation_height / { tan( 0.5*camera_angle_vertical * ( obj_y2 - (frame_height / 2)) / ( frame_height / 2 ))}
        '''
        cat_ind = self.cat_ind
        sizes_dict = KNOWN_CLASS_SIZES
        obj_width = self.y2 - self.y1
        obj_height = self.x2 - self.x1
        frame_width = CAM_LENS_INFO['h_pixels']
        frame_height = CAM_LENS_INFO['v_pixels']
        width_fov = CAM_LENS_INFO['h_fov'],
        height_fov = CAM_LENS_INFO['v_fov'],
        width_fov = width_fov[0] * (math.pi / 180) #deg to rad
        height_fov = height_fov[0] * (math.pi / 180)  # deg to rad
        cam_installation_height = CAM_LENS_INFO['install_height']

        d_pred_width = smart_divide(sizes_dict[cat_ind][0], math.tan(width_fov * (obj_width/frame_width)))
        #d_pred_width = (sizes_dict[cat_ind][0])/(math.tan(width_fov * (obj_width/frame_width)))

        d_pred_height = smart_divide(sizes_dict[cat_ind][1], math.tan(height_fov * (obj_height/frame_height)))
        #d_pred_height = (sizes_dict[cat_ind][1])/(math.tan(height_fov * (obj_height/frame_height)))

        # === calculating distance based on y2 is not useful
        #if self.y2 > frame_height/2:
        #    d_pred_bottom = cam_installation_height / ( math.tan( 0.5*height_fov * ( self.y2 - frame_height/2 ) / ( frame_height/2 )))
        #else: # above horizon
        #    d_pred_bottom = None
        d_pred_bottom = None

        distance_predictions = [d_pred_width, d_pred_height, d_pred_bottom]
        enabled_preds = [x for x in distance_predictions if x is not None]
        avg_d_pred = sum(enabled_preds)/len(enabled_preds)

        return avg_d_pred

def smart_divide(a,b):
    return a/b if b != 0 else a/(b+1)
