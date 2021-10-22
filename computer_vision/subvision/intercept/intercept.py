import sys
sys.path.append('../')
from utils import CameraFacing

import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KalmanPointObject:
    '''
    This class represents the internal state of individual tracked points on a x,y grid.
    '''
    def __init__(self, category_id, camera_facing, init_point=np.transpose(np.array([40,1]))):
        self.category_id = category_id
        self.camera_facing = camera_facing
        self.birth_location = init_point # used to avoid warning in group riding / paceline situations; only warn about people approach from a distance
        self.hit_streak = 0 # used as a dumb filter
        self.warned_about = False

        # important calculated features
        self.time_to_encounter = 9999
        self.encounter_proximity = 9999


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
            [500.,0,0,0], # 500 is set high for initial x value
            [0.,  9,0,0] #9 is from (max x velocity/3)^2 in m/s.
            [0.,0,100,0], # initial y value variance is a bit lower than that for x
            [0.,0,  0,1]]) # initial y velocity variance is also lower

        # state transition function: old state is multiplied by this array to get naive prediction of new state
        dt = 0.03 # time step is 30 frames/sec
        self.kf.F = np.array([
            [1.,dt,0, 0], # new x value = prev_x_value * 1 + prev_x_velocity*dt
            [0.,1, 0, 0], # new_x_velocity = old_x_velocity (constant velocity model)
            [0.,0, 1,dt], # same but for y
            [0.,0, 0, 1])

        # Process noise: external factors that alter kinematics of object out of the expected Newtonian mechanics
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=0.03, var=0.5)
        # dt is 30 fps
        # var is a guess

        # Control Function: if you can influence the thing being measured, use this.
        self.kf.B = None

        self.kf.x = init_point

    def update(self, coordinates):
        kf_point.predict()
        kf_point.update(coordinates)
        x_pred = kf_point.x

        # TODO pseudocode
        x = x_pred[0]
        vx = x_pred[1]
        y = x_pred[2]
        vy = x_pred[3]

        self.time_to_encounter = x / vx
        self.encounter_proximity = vy * self.time_to_encounter



class Intercept:
    '''
    Create and manage KalmanPointObjects
    '''
    def __init__(self):
        self.kf_points= [] # a list of dictionaries, where key is track_id and value is a KalmanPointObject
        self.front_category_tte = {
            # category id : time to encounter threshold (seconds)
            0 : 3.,
            1 : 3.,
            2 : 3.,
            3 : 3.,
            4 : 3.,
            5 : 3.,
            6 : 3.,
            }
        self.rear_category_tte= {
            # category id : time to encounter threshold (seconds)
            0 : 3.,
            1 : 3.,
            2 : 3.,
            3 : 3.,
            4 : 3.,
            5 : 3.,
            6 : 3.,
            }
        self.front_category_ep = {
            # category id : encounter proximity threshold (meters)
            0 : 3.,
            1 : 3.,
            2 : 3.,
            3 : 3.,
            4 : 3.,
            5 : 3.,
            6 : 3.,
            }
        self.rear_category_ep = {
            # category id : encounter proximity threshold (meters)
            0 : 3.,
            1 : 3.,
            2 : 3.,
            3 : 3.,
            4 : 3.,
            5 : 3.,
            6 : 3.,
            }

        self.front_warning = None
        self.rear_warning = None


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

        an np.array of shape (n,14) where
        output:
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
            11 : y coordinate
        NEW 12 : time to encounter
        NEW 13 : y-intercept of encounter
        '''

        clean_input = self.preprocess(input)

        '''
            0 : track id
            1 : x coordinate
            2 : y coordinate
            3 : class index a.k.a category_id
            4 : assigned to front (0) or rear (1)
        NEW 5 : empty (time to encounter)
        NEW 6 : empty (encounter y-intercept)
        '''

        # update or create new kalman filter object
        for det in clean_input:
            track_id = det[0]
            x_coord = det[1]
            y_coord = det[2]
            coordinates = np.transpose(np.array([x_coord, y_coord])) # transposed array for pykalman input
            category_id = dets[3]
            if det[4] == 0:
                camera_facing = CameraFacing.FRONT
            else:
                camera_facing = CameraFacing.REAR

            if track_id in self.kf_points.keys():
                kf_point = self.kf_points[track_id]
                kf_point_state = kf_point.update(coordinates)
            else:
                self.kf_points.append(
                        {track_id : KalmanPointObject(camera_facing, category_id, coordinates)}
                        )

        # check if any meet condition for warning
        for point_key in self.kf_points:
            point_obj = self.kf_points[point_key]

            if self.kf_points[point_key].camera_facing == CameraFacing.FRONT:
                front_TTE_warning = point_obj.time_to_encounter < self.front_category_tte[point_obj.category_id]
                front_EP_warning = point_obj.encounter_proximity < self.front_catgory_ep[point_obj.category_id]

            else: # CamFacing.REAR
                rear_TTE_warning = point_obj.time_to_encounter < self.rear_category_tte[point_obj.category_id]
                rear_EP_warning = point_obj.encounter_proximity < self.rear_catgory_ep[point_obj.category_id]

            if front_TTE_warning and front_EP_warning:
                self.front_warning = True

            if rear_TTE_warning and rear_EP_warning:
                self.rear_warning = True
        return self.front_warning, self.rear_warning

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



