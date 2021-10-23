from ..utils import CameraFacing
from PARAMETERS import FRONT_TTE_THRESH, REAR_TTE_THRESH, FRONT_EP_THRESH, REAR_EP_THRESH, FRONT_HIT_STREAK_THRESH, REAR_HIT_STREAK_THRESH

import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KalmanPointObject:
    '''
    This class represents the internal state of individual tracked points on a x,y grid.

    '''

    def __init__(self, track_id, camera_facing, category_id, init_point=np.array([40,1])):
        # init point should be [x_position, x_velocity, y_position, y_velocity]
        self.track_id = track_id
        self.category_id = category_id
        self.camera_facing = camera_facing
        self.birth_location = init_point # used to avoid warning in group riding / paceline situations; only warn about people approach from a distance

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # important calculated features
        self.time_to_encounter = None# time_to_encounter(tte)
        self.encounter_proximity = None# encounter_proximity (ep)

        if self.camera_facing == CameraFacing.FRONT:
            self.tte_thresh = FRONT_TTE_THRESH[self.category_id]
            self.ep_thresh = FRONT_EP_THRESH[self.category_id]
        else: # CameraFacing.REAR
            self.tte_thresh = REAR_TTE_THRESH[self.category_id]
            self.ep_thresh = REAR_EP_THRESH[self.category_id]

        self.over_threshold = []
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
        self.kf.P = np.array([
            [100.,0,0,0], # 500 is set high for initial x value
            [0.,5,0,0], #9 is from (max x velocity/3)^2 in m/s.
            [0.,0,10,0], # initial y value variance is a bit lower than that for x
            [0.,0,0,0.1]]) # initial y velocity variance is also lower

        # state transition function: old state is multiplied by this array to get naive prediction of new state
        dt = 0.03 # time step is 30 frames/sec
        self.kf.F = np.array([
            [1.,dt,0, 0], # new x value = prev_x_value * 1 + prev_x_velocity*dt
            [0.,1, 0, 0], # new_x_velocity = old_x_velocity (constant velocity model)
            [0.,0, 1,dt], # same but for y
            [0.,0, 0, 1]])

        # Process noise: external factors that alter kinematics of object out of the expected Newtonian mechanics
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=0.03, var=10)
        # dt is 30 fps
        # var is a guess

        # Control Function: if you can influence the thing being measured, use this.
        self.kf.B = None

        self.kf.z = init_point

    def update(self, coordinates):

        self.kf.predict()
        self.kf.update(coordinates)
        x_pred = self.kf.x

        x = x_pred[0]
        dxdt = x_pred[1]
        y = x_pred[2]
        dydt = x_pred[3]

        # KEY FEATURES
        self.time_to_encounter = - x / dxdt
        self.encounter_proximity = dydt * self.time_to_encounter # assuming linear motion

        # threshold constants are defined with respect to 'left/right' of frame
        # a x,y coordinate system is used here.
        # therefore some signs need flipping
        if self.camera_facing == CameraFacing.FRONT:
            upper_ep_bound = self.ep_thresh[0]
            lower_ep_bound = self.ep_thresh[1] * -1
        else: # CameraFacing.REAR
            lower_ep_bound = self.ep_thresh[0] * -1
            upper_ep_bound = self.ep_thresh[1]

        ep_thresh_violated = (self.encounter_proximity > lower_ep_bound) and (self.encounter_proximity < upper_ep_bound)

        if (self.time_to_encounter < self.tte_thresh) and self.time_to_encounter > 0 and ep_thresh_violated:
            self.over_threshold.append((self.time_to_encounter, self.encounter_proximity))

        if self.camera_facing == CameraFacing.FRONT:
            if len(self.over_threshold) == FRONT_HIT_STREAK_THRESH:
                self.dangerous = True
        else:
            if len(self.over_threshold) == REAR_HIT_STREAK_THRESH:
                self.dangerous = True



class Intercept:
    '''
    Create and manage KalmanPointObjects

    At every update cycle, KalmanPointObjects calculates two key features:
    1. Time to Encounter (TTE): Predicted time (in seconds) until overtake occurs.
    2. Encounter Proximity (PE): Predicted distance between object and myself when overtake occurs.

    KalmanPointObjects whose above values are within a threshold are considered 'dangerous'.

    '''
    def __init__(self,
            max_age=2
            ):
        self.kf_points= {} # a dict, where key is track_id and value is a KalmanPointObject
        self.max_age = 2


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

        # update or create new kalman filter object
        for det in clean_input:
            track_id = det[0]
            x_coord = det[1]
            y_coord = det[2]
            coordinates = np.array([x_coord, y_coord]) # np.array([x_pos, x_velo, y_pos, y_velo])
            category_id = det[3]
            if det[4] == 0:
                camera_facing = CameraFacing.FRONT
            else:
                camera_facing = CameraFacing.REAR

            if track_id in self.kf_points:
                kf_point = self.kf_points[track_id]
                kf_point_state = kf_point.update(coordinates)
            else:
                #
                self.kf_points.update(
                        {track_id : KalmanPointObject(track_id, camera_facing, category_id, coordinates)}
                        )


        # TODO delete old kalman filter objects
        for track_id in self.kf_points:
            pass




        # check if any point objects are posing immediate threat
        front_dangers = {}
        rear_dangers = {}
        for point_key in self.kf_points:
            point = self.kf_points[point_key]

            if point.dangerous == True:
                if point.camera_facing == CameraFacing.FRONT:
                    front_dangers.update({int(point.track_id) : (point.time_to_encounter, point.encounter_proximity)})
                else:
                    rear_dangers.update({int(point.track_id) : (point.time_to_encounter, point.encounter_proximity)})
        return front_dangers, rear_dangers

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

        an np.array of shape(n,7) where
            0 : track id
            1 : x coordinate
            2 : y coordinate
            3 : class index
            4 : assigned to front (0) or rear (1)
        NEW 5 : empty (time to encounter)
        NEW 6 : empty (encounter y-intercept)
        '''
        clean_inp = np.zeros((inp.shape[0], 7))
        clean_inp[:,(0,1,2,3,4)] = inp[:,(8,10,11,4,7)]

        return clean_inp



