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


# Time-to-encounter and Encounter-proximity thresholds for each category

FRONT_TTE_THRESH = {
    # category id : time to encounter threshold (seconds)
    0 : 3.,
    1 : 3.,
    2 : 3.,
    3 : 3.,
    4 : 3.,
    5 : 3.,
    6 : 3.,
    }
REAR_TTE_THRESH = {
    # category id : time to encounter threshold (seconds)
    0 : 3.,
    1 : 3.,
    2 : 3.,
    3 : 3.,
    4 : 3.,
    5 : 3.,
    6 : 3.,
    }
FRONT_EP_THRESH= {
    # category id : encounter proximity threshold (meters)
    0 : 5.,
    1 : 5.,
    2 : 5.,
    3 : 5.,
    4 : 5.,
    5 : 5.,
    6 : 5.,
    }
REAR_EP_THRESH= {
    # category id : encounter proximity threshold (meters)
    0 : 5.,
    1 : 5.,
    2 : 5.,
    3 : 5.,
    4 : 5.,
    5 : 5.,
    6 : 5.,
    }

# Hit streak threshold (TTE and EQ are both true)

FRONT_HIT_STREAK_THRESH = 5
REAR_HIT_STREAK_THRESH = 3
