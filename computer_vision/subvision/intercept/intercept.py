import time

import numpy as np

from ..utils import CameraFacing
from PARAMETERS import *

from .pyplot_vis import plot_birds_eye

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KalmanPointObject:
    '''
    This class represents the internal state of individual tracked points on a x,y grid.

    '''

    def __init__(self, track_id, camera_facing, category_id, init_point, x_center, y_center):
        # init point should be [x_position, y_position ]
        self.track_id = track_id
        self.category_id = category_id
        self.camera_facing = camera_facing
        self.birth_location = init_point # used to avoid warning in group riding / paceline situations; only warn about people approach from a distance

        # not used in this class, but later for curtain filter
        self.x_center = x_center
        self.y_center = y_center

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # intermediate kalman filter predictions, used internally but also exposed as an instance variable for visualization.
        self.filter_pred_x = 0
        self.filter_pred_dxdt = 0
        self.filter_pred_y = 0
        self.filter_pred_dydt = 0

        # important calculated features
        self.time_to_encounter = None# time_to_encounter(tte)
        self.encounter_proximity = None# encounter_proximity (ep)

        if self.camera_facing == CameraFacing.FRONT:
            self.tte_thresh = FRONT_TTE_THRESH[self.category_id]
            self.ep_thresh = FRONT_EP_THRESH[self.category_id]
        else: # CameraFacing.REAR
            self.tte_thresh = REAR_TTE_THRESH[self.category_id]
            self.ep_thresh = REAR_EP_THRESH[self.category_id]

        self.over_threshold = 0
        # to be filled with tuple (time_to_encounter, encounter_proximity) if both thresholds are reached at every frame

        self.dangerous = False

        # FOR TUTORIAL ON KALMAN FILTER, see
        # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # 4 variables in total
        # 2 measurements (sensors) (x,y inputs are separate)

        # measurement function
        # take the prior, convert it to a measurement by multiplying it with H,
        # and subtract that from the measurement
        # in this case, we have x,y position.
        # z is a two-variable array
        # z = [[z1], [z2]] shape (2,1)
        # the residual equation has the form y=z-Hx (y is the residual, x is the prior)
        # since shape of x is (4,1)
        # shape of H: (2,4)
        self.kf.H = np.array([
            [1.,0,0,0],
            [0.,0,1,0]])

        # variance of measurement errors for each reading sensor
        # the diagonals are variances
        # non-diagonals are used if there is a correlation between different sensors
        self.kf.R = np.array([
            [10., 0],
            [0., 10]])

        # state covariance
        x_v = KPO_X_VAR
        xd_v = KPO_X_DOT_VAR
        y_v = KPO_Y_VAR
        yd_v = KPO_Y_DOT_VAR

        self.kf.P = np.array([
            [x_v, 0.,0,0], # Unknown initial x value -> large number (1000)
            [0.,xd_v,0,0], #9 is from (max x velocity/3)^2 in m/s.
            [0.,0, y_v,0], # initial y value variance is a bit lower than that for x
            [0.,0,0,yd_v]]) # initial y velocity variance is also lower

        # state transition function: old state is multiplied by this array to get naive prediction of new state
        dt = 0.03 # time step is 30 frames/sec
        self.kf.F = np.array([
            [1.,dt,0, 0], # new x value = prev_x_value * 1 + prev_x_velocity*dt
            [0.,1, 0, 0], # new_x_velocity = old_x_velocity (constant velocity model)
            [0.,0, 1,dt], # same but for y
            [0.,0, 0, 1]])

        # Process noise: external factors that alter kinematics of object out of the expected Newtonian mechanics
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=0.03, var=100)
        # dt is 30 fps
        # var is a guess

        # Control Function: if you can influence the thing being measured, use this.
        self.kf.B = None

        self.kf.z = init_point

    def update(self, coordinates, x_center, y_center):
        '''
        encounter_proximity (EP) signs explained:
        ┌────────────────────────────────┐
        │             Front              │
        │                                │
        │               ┼                │
        │  ◄──────────     ───────────►  │
        │  positive EP  ┼  negative EP   │
        │                                │
        │               ┼                │
        │                                │
        ├────────────────────────────────┤
        │             Rear               │
        │               ┼                │
        │                                │
        │  ◄─────────── ┼  ───────────►  │
        │  negative EP     positive EP   │
        │               ┼                │
        │                                │
        │               │                │
        └───────────────┴────────────────┘
        '''
        self.x_center = x_center
        self.y_center = y_center


        self.kf.predict()
        self.kf.update(coordinates)
        x_pred = self.kf.x

        # these instance variables are used for visualization
        # they are compared to the raw inputs, to see what effect the kalman filter has
        self.filter_pred_x = x_pred[0]
        self.filter_pred_dxdt = x_pred[1]
        self.filter_pred_y = x_pred[2]
        self.filter_pred_dydt = x_pred[3]

        x = x_pred[0]
        dxdt = x_pred[1]
        y = x_pred[2]
        dydt = x_pred[3]

        # KEY FEATURES
        self.time_to_encounter = - x / dxdt
        self.encounter_proximity = dydt * self.time_to_encounter # assuming linear motion

        # see above diagram for explanation of signs
        if self.camera_facing == CameraFacing.FRONT:
            upper_ep_bound = self.ep_thresh[0]
            lower_ep_bound = self.ep_thresh[1] * -1
        else: # CameraFacing.REAR
            lower_ep_bound = self.ep_thresh[0] * -1
            upper_ep_bound = self.ep_thresh[1]

        ep_thresh_violated = (self.encounter_proximity > lower_ep_bound) and (self.encounter_proximity < upper_ep_bound)
        tte_thresh_violated = (self.time_to_encounter < self.tte_thresh) and (self.time_to_encounter > 0)

        if (tte_thresh_violated and ep_thresh_violated):
            self.over_threshold += 1 * HIT_MULTIPLIER
        else:
            self.over_threshold -= 1 * MISS_MULTIPLIER

        if self.camera_facing == CameraFacing.FRONT:
            if self.over_threshold > FRONT_HIT_STREAK_THRESH:
                self.dangerous = True
                self.over_threshold = min(self.over_threshold, FRONT_MAX_STREAK)
            else:
                self.dangerous = False
                self.over_threshold = max(0, self.over_threshold) # no negative count of hits
        else:
            if self.over_threshold > REAR_HIT_STREAK_THRESH:
                self.dangerous = True
                self.over_threshold = min(self.over_threshold, REAR_MAX_STREAK)
            else:
                self.dangerous = False
                self.over_threshold = max(0,self.over_threshold)

class Intercept:
    '''
    Create and manage KalmanPointObjects

    At every update cycle, KalmanPointObjects calculates two key features:
    1. Time to Encounter (TTE): Predicted time (in seconds) until overtake occurs.
    2. Encounter Proximity (PE): Predicted distance between object and myself when overtake occurs.

    KalmanPointObjects whose above values are within a threshold are considered 'dangerous'.

    '''
    def __init__(self, save_vis=False, view_vis=False, frame_width=640
            ):
        self.save_vis = save_vis
        self.view_vis = view_vis
        self.frame_width = frame_width
        self.kf_points= {} # a dict, where key is track_id and value is a KalmanPointObject

        # objects that are about to overtake me; live updated
        self.front_dangers = {}
        self.rear_dangers = {}

        self.front_last_ring_time = 0
        self.rear_last_ring_time = 0

        self.already_warned_ids = [] # do not emit multiple rings for the same object

        # 'Curtain': an extremely simple, "don't ring if bbox is not within a certain horizontal range".

        self.front_curtain_limits = (0.4, 1.0) # (left limit, right limit) where 0 is the left edge and 1 is the right edge.
        self.rear_curtain_limits = (0.0, 1.0)




    def curtain_filter(self, front_dangers, rear_dangers):
        '''
        Ignore 'dangerous' bounding boxes if they are out of the pre-defined horizontal range.
        This is optional. It reduces false positives (when people are coming at me from the other side of the road, instead of me passing them.)
        Args:
            camera_facing: choose between CameraFacing.FRONT or CameraFacing.REAR enum objects.

        Illustration:

            Two objects are approaching me, but only the right one triggers alert.
            In this example, the curtain limits are (0.4, 1.0)

                        0.4
        0.0              │                      1.0
         ┌───────────────┼───────────────────────┐
         │...............│                       │
         │...............│                       │
         │...............│                       │
         │...............│                       │
         │...............│                       │
         │...┌───────┐...│  x|x   ┌──────┐       │
         │...│.......│...│xxx|xxx │      │       │
         │...│.......│. x│xxx|xxxx│warned│       │
         │...│.......│xxx│xxx|xxxx│      │       │
         │...│ignored│xxx│xxx|xxxx│      │       │
         │...│.......│xxx│xxx|xxxx└──────┘x      │
         │...└───────┘xxx│xxx|xxxxxxxxxxxxxxxx   │
         │ xxxxxxxxxxxxxx│xxx|xxxxxxxxxxxxxxxxxxx│
         └───────────────┼───────────────────────┘
                         │
        '''

        filtered_front_dangers = {}
        filtered_rear_dangers = {}

        for key,value in front_dangers.items():
            x_center = value['xc']
            normalized_x_center = x_center / self.frame_width
            if not ((normalized_x_center <= self.front_curtain_limits[0]) or (normalized_x_center >= self.front_curtain_limits[1])):
                filtered_front_dangers.update({key:value})
        for key,value in rear_dangers.items():
            x_center = value['xc']
            normalized_x_center = x_center / self.frame_width
            if not ((normalized_x_center <= self.rear_curtain_limits[0]) or (normalized_x_center >= self.rear_curtain_limits[1])):
                filtered_rear_dangers.update({key:value})

        return filtered_front_dangers, filtered_rear_dangers




    def step(self, input):
        '''
        expected input:

        an np.array of shape(n,12) where
            0 : x1
            1 : y1
            2 : x2
            3 : y2
            4 : class index
            5 : x_center
            6 : y_center
            7 : assigned to front (0) or rear(1)
            8 : track id
            9 : distance
            10 : x coordinate
            11: y coordinate

        '''

        clean_input = self.preprocess(input)
        '''
        clean_input format:

        an np.array of shape(n,9) where
            0 : track id
            1 : x coordinate
            2 : y coordinate
            3 : class index
            4 : assigned to front (0) or rear (1)
        NEW 5 : empty (time to encounter)
        NEW 6 : empty (encounter y-intercept)
            7 : x_center
            8 : y_center
        '''

        # update or create new kalman filter object
        for det in clean_input:
            track_id = det[0]
            x_coord = det[1]
            y_coord = det[2]
            x_center = det[7]
            y_center = det[8]
            coordinates = np.array([x_coord, y_coord]) # np.array([x_pos, x_velo, y_pos, y_velo])
            category_id = det[3]
            if det[4] == 0:
                camera_facing = CameraFacing.FRONT
            else:
                camera_facing = CameraFacing.REAR

            if track_id in self.kf_points:
                kf_point = self.kf_points[track_id]
                kf_point_state = kf_point.update(coordinates, x_center, y_center)
            else:
                #
                self.kf_points.update(
                        {track_id : KalmanPointObject(track_id, camera_facing, category_id, coordinates, x_center, y_center)}
                        )

        # delete some old kfpoint objects in history
        # as long as not saving visualizations (longer history required for visualization)
        if not (self.save_vis or self.view_vis) and len(self.kf_points) > 100:
            for key_del in sorted(self.kf_points.keys())[:50]: #delete the 50 oldest ones
                self.kf_points.pop(key_del)


        # check if any point objects are posing immediate threat
        for point_key in self.kf_points:
            point = self.kf_points[point_key]

            if point.dangerous == True:
                new_entry = {int(point.track_id) :
                        {
                            'tte':point.time_to_encounter,
                            'ep': point.encounter_proximity,
                            'xc': point.x_center,
                            'yc': point.y_center,
                        }
                    }
                if point.camera_facing == CameraFacing.FRONT:
                    self.front_dangers.update(new_entry)
                else:
                    self.rear_dangers.update(new_entry)

            else: #point.dangerous == False
                if point.camera_facing == CameraFacing.FRONT:
                    if int(point.track_id) in self.front_dangers.keys():
                        self.front_dangers.pop(point.track_id)
                else:
                    if int(point.track_id) in self.rear_dangers.keys():
                        self.rear_dangers.pop(point.track_id)

        # VISUALIZATION ======================================================================================

        # get track_ids of current points detected (self.kf_points contains a bunch of dead tracks)
        plot_image = None
        if self.save_vis or self.view_vis:

            current_active_track_ids = [row[0] for row in clean_input]
            birds_eye_vis_data = {}

            # populate using raw data, which contains the current list of actively seen detections
            for row in clean_input:
                birds_eye_vis_data.update(
                        {row[0]:
                            {
                            'raw_x':row[1],
                            'raw_y':row[2],
                            'class_index':row[3],
                            'TTE':row[5],
                            'EP':row[6],
                            }
                            }
                    )

            # populate the kalman filter data (use only currently seen detections from above)
            for track_id in current_active_track_ids:
                kf_point = self.kf_points[track_id]
                birds_eye_vis_data[track_id].update({
                    'filter_pred_x':kf_point.filter_pred_x[0] if kf_point.filter_pred_x !=0 else 999,
                    'filter_pred_y':kf_point.filter_pred_y[0] if kf_point.filter_pred_y !=0 else 999,
                    'filter_pred_dxdt':kf_point.filter_pred_dxdt[0] if kf_point.filter_pred_dxdt !=0 else 999,
                    'filter_pred_dydt':kf_point.filter_pred_dydt[0] if kf_point.filter_pred_dydt !=0 else 999,
                    'dangerous':kf_point.dangerous,
                    'TTE':kf_point.time_to_encounter,
                    'EP':kf_point.encounter_proximity,
                    })


            #plot_birds_eye(birds_eye_vis_data)
            plot_image = plot_birds_eye(birds_eye_vis_data, save_every_plot=self.save_vis)
        # end VISUALIZATION ======================================================================================



        return self.front_dangers, self.rear_dangers, plot_image

    def should_ring_now(self):
        '''
        Ring only if we haven't already alerted this person,
        and only ring every MIN_RING_GAP seconds.
        '''
        front_ring_now = False
        rear_ring_now = False

        for key in self.front_dangers:
            if key in self.already_warned_ids:
                continue
            else:
                if time.time() - self.front_last_ring_time > FRONT_MIN_RING_GAP:
                    front_ring_now = True
                    self.already_warned_ids.append(key)
                    self.front_last_ring_time = time.time()

        for key in self.rear_dangers:
            if key in self.already_warned_ids:
                continue
            else:
                if time.time() - self.front_last_ring_time > REAR_MIN_RING_GAP:
                    rear_ring_now = True
                    self.already_warned_ids.append(key)
                    self.rear_last_ring_time = time.time()

        return front_ring_now, rear_ring_now


    def preprocess(self, inp):
        '''
        expected input:

        an np.array of shape(n,12) where
            0 : x1
            1 : y1
            2 : x2
            3 : y2
            4 : class index
            5 : x_center
            6 : y_center
            7 : assigned to front (0) or rear(1)
            8 : track id
            9 : distance
            10 : x coordinate
            11: y coordinate

        sanitized output:

        an np.array of shape(n,9) where
            0 : track id
            1 : x coordinate
            2 : y coordinate
            3 : class index
            4 : assigned to front (0) or rear (1)
        NEW 5 : empty (time to encounter)
        NEW 6 : empty (encounter y-intercept)
            7 : x_center
            8 : y_center
        '''
        clean_inp = np.zeros((inp.shape[0], 9))
        clean_inp[:,(0,1,2,3,4,7,8)] = inp[:,(8,10,11,4,7,5,6)]

        return clean_inp



