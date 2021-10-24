# Class index to name
HYDO_CLASSES = ['bicycle', 'bus', 'car', 'cyclist', 'motorcycle', 'pedestrian', 'truck'] # alphabetically ordered because that's how it's trained


# GUESSTIMATIONS
KNOWN_CLASS_SIZES = { # as seen from front or back
        # category id : (width (meters), height (meters))
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
        'h_fov' : 60, # horizontal field of view
        'v_fov' : 33.75, # vertical field of view
        'h_pixels' : 960,
        'v_pixels' : 540,
        'install_height': 0.8 # (meters)
        }

# Optional: if camera is not centered on horizon, use this to translate the picture
FRONT_NUDGE_DOWN = -150 # negative numbers for nudge frame upwards
FRONT_NUDGE_RIGHT = 0
REAR_NUDGE_DOWN = 0
REAR_NUDGE_RIGHT = 0


# Time-to-encounter and Encounter-proximity thresholds for each category


# disable any category detection by setting value to negative value
FRONT_TTE_THRESH = {
    # category id : time to encounter threshold (seconds)
    0 : 10., # bicycle
    1 : 10., # bus
    2 : 10., # car
    3 : 10., # cyclist
    4 : 10., # motorcycle
    5 : 10., # pedestrian
    6 : 10., # truck
    }
REAR_TTE_THRESH = {
    # category id : time to encounter threshold (seconds)
    0 : 10., # bicycle
    1 : 15., # bus
    2 : 15., # car
    3 : 10., # cyclist
    4 : 10., # motorcycle
    5 : 10., # pedestrian
    6 : 15., # truck
    }
FRONT_EP_THRESH= {
    # category_id : (left_proximity, right_proximity)
    # in meters
    0 : ( 1., 3.),
    1 : ( 1., 3.),
    2 : ( 1., 3.),
    3 : ( 1., 3.),
    4 : ( 1., 3.),
    5 : ( 1., 3.),
    6 : ( 1., 3.),
    }
REAR_EP_THRESH= {
    # category_id : (left_proximity, right_proximity)
    # in meters
    0 : ( 5., 5.),
    1 : ( 5., 5.),
    2 : ( 5., 5.),
    3 : ( 5., 5.),
    4 : ( 5., 5.),
    5 : ( 5., 5.),
    6 : ( 5., 5.),
    }

# Hit streak threshold (TTE and EQ are both true)

FRONT_HIT_STREAK_THRESH = 50
REAR_HIT_STREAK_THRESH = 5

FRONT_MAX_STREAK = 80 # If object is not a threat for consecutive frames, object will still stay in danger status for (FRONT_MAX_STREAK - FRONT_HIT_STREAK_THRESH) / (MISS_MULTIPLIER) frames
REAR_MAX_STREAK = 80



# If HIT_mulitplier > MISS_multiplier, a lot of misses will be required to 'deactivate' the Danger status of object.
HIT_MULTIPLIER = 5 # number of consecutive frames of hits to trigger danger status = FRONT(REAR)_HIT_STREAK_THRESH / HIT_MULTIPLIER
MISS_MULTIPLIER = 1 # number of consecutive misses to downgrade danger status = FRONT(REAR)_MAX_STREAK / MISS_MULTIPLIER

# KalmanPointObject (KPO) Parameters

KPO_X_VAR = 1000
KPO_X_DOT_VAR = 1
KPO_Y_VAR = 1000
KPO_Y_DOT_VAR = 0.1

# Simple Online Realtime Tracker (SORT) parameters

## Kalman filter initialization parameters

SORT_R_MULT = 10. # variance of measurement error (set to high if object detector isn't good)
SORT_P1_MULT = 1000. # variance of state. Is set to high for unknown initial velocities
SORT_P2_MULT = 10. # variance of state (besides initial velocites)
SORT_Q_MULT = 0.1 # process noise

## Tracker parameters

SORT_MAX_AGE = 1000
SORT_MIN_HITS = 2
SORT_IOU_THRESH = 0.2

# Audio parameters
FRONT_MIN_RING_GAP = 3 # avoid overlapping audio alarm
REAR_MIN_RING_GAP = 3
