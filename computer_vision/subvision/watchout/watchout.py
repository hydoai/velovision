import numpy as np
import math
import os

# GUESSTIMATIONS
# category : width(meters) * height(meters)
KNOWN_CLASS_SIZES = { # as seen from front or back
        0 : (0.3,1), # bicycle
        1 : (3.5,3.2), # bus
        2 : (2,2), # car
        3 : (0.6,1.7), # cyclist
        4 : (0.8,1.7), # motorcycle
        5 : (0.5,1.7), # pedestrian
        6 : (3.5,4), # truck
        }

# GUESSTIMATIONS
CAM_LENS_INFO = {
        'h_fov' : 45, # horizontal field of view
        'v_fov' : 25, # vertical field of view
        'h_pixels' : 960, 
        'v_pixels' : 540,
        'install_height': 0.8 # (meters)
        }

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
        self.memory_earliest_remaining_index = 0

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
            9 : danger_rating (0-none, 10-max)

        '''
        for obj in tracked_dets:
            if obj[8].item() in self.memory.keys(): #we've seen this guy before
                watchedthing = self.memory[obj[8].item()]
                watchedthing.step(obj)
                print(f"Predicted distance: {watchedthing.pred_distance}m")

            else:
                self.memory.update({obj[8].item(): WatchedThing(obj)}) # create new Thing

                # if memory contains more than 100, delete 10 earliest entries
                if len(self.memory) > 100:
                    for i in range(self.memory_earliest_remaining_index, self.memory_earliest_remaining_index + 10):
                        del self.memory[i]
                    self.memory_earliest_remaining_index += 10

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

        d_pred_width = sizes_dict[cat_ind][0] / ( math.tan( width_fov * ( obj_width / frame_width ) ) )
        d_pred_height = sizes_dict[cat_ind][1] / ( math.tan( height_fov * ( obj_height / frame_height ) ) )

        if self.y2 > frame_height/2:
            d_pred_bottom = cam_installation_height / ( math.tan( 0.5*height_fov * ( self.y2 - frame_height/2 ) / ( frame_height/2 )))
        else: # above horizon
            d_pred_bottom = 9999999999999999999

        distance_predictions = [d_pred_width, d_pred_height, d_pred_bottom]
        enabled_preds = [x for x in distance_predictions if x is not None]
        avg_d_pred = sum(enabled_preds)/len(enabled_preds)

        return avg_d_pred

