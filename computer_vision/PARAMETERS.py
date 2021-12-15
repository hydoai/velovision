# Class index to name
# alphabetically ordered because that's how it's trained
HYDO_CLASSES = ['bicycle', 'bus', 'car',
                'cyclist', 'motorcycle', 'pedestrian', 'truck']


# GUESSTIMATIONS
KNOWN_CLASS_SIZES = {  # as seen from front or back
    # category id : (width (meters), height (meters))
    0: (0.3, 1),  # bicycle
    1: (3.5, 3.2),  # bus
    2: (2, 2),  # car
    3: (0.6, 1.7),  # cyclist
    4: (0.8, 1.7),  # motorcycle
    5: (0.5, 1.7),  # pedestrian
    6: (3.5, 4),  # truck
}

UNIFIED_HEIGHT = 360
UNIFIED_WIDTH = 640

# HYDO DevKit-One Front/Rear Camera Specifications
class CameraInfo:
    def __init__(self,
                 width_res,
                 height_res,
                 width_fov,
                 height_fov,
                 fps,
                 install_height,
                 inference_crop_ratio,
                 nudge_down=0,
                 nudge_right=0,
                 ):
        self.width_res = width_res
        self.height_res = height_res
        self.width_fov = width_fov
        self.height_fov = height_fov
        self.fps = fps
        self.install_height = install_height
        self.inference_crop_ratio = inference_crop_ratio # for simplicity, ratio is based on width only, in case the output image ratio is not the same as the input image ratio.
        self.nudge_down = nudge_down # negative numbers for nudge up
        self.nudge_right = nudge_right #negative numbers for nudge left 

DK1_frontcam = CameraInfo(
    width_res=1280,
    height_res=720,
    width_fov=60,
    height_fov=45,
    fps=30,
    install_height=0.8,
    inference_crop_ratio = 2
)

DK1_rearcam = CameraInfo(
    width_res=640,
    height_res=480,
    width_fov=60,
    height_fov=45,
    fps=30,
    install_height=0.8,
    inference_crop_ratio = 2
)

GoProHD_front = CameraInfo(
    width_res = 1920,
    height_res = 1080,
    width_fov = 90,
    height_fov = 60,
    fps=30,
    install_height=0.8,
    inference_crop_ratio = 1,
    nudge_down= -150
)

GoProHD_rear = CameraInfo(
    width_res = 1920,
    height_res = 1080,
    width_fov = 90,
    height_fov = 60,
    fps=30,
    install_height=0.8,
    inference_crop_ratio = 1 
)


# Time-to-encounter and Encounter-proximity thresholds for each category
# disable any category detection by setting value to negative value
FRONT_TTE_THRESH = {
    # category id : time to encounter threshold (seconds)
    0: -1.,  # bicycle
    1: 10.,  # bus
    2: 10.,  # car
    3: 10.,  # cyclist
    4: 10.,  # motorcycle
    5: -1,  # pedestrian
    6: 10.,  # truck
}
REAR_TTE_THRESH = {
    # category id : time to encounter threshold (seconds)
    0: -1.,  # bicycle
    1: 10.,  # bus
    2: 10.,  # car
    3: 10.,  # cyclist
    4: 10.,  # motorcycle
    5: -1.,  # pedestrian
    6: 10.,  # truck
}
FRONT_EP_THRESH = {
    # category_id : (left_proximity, right_proximity)
    # in meters
    0: (1., 5.),
    2: (1., 5.),
    1: (1., 5.),
    3: (1., 5.),
    5: (1., 5.),
    6: (1., 5.),
    4: (1., 5.),
}
REAR_EP_THRESH = {
    # category_id : (left_proximity, right_proximity)
    # in meters
    1: (5., 5.),
    0: (5., 5.),
    2: (5., 5.),
    3: (5., 5.),
    4: (5., 5.),
    5: (5., 5.),
    6: (5., 5.),
}

# Hit streak threshold (TTE and EQ are both true)

FRONT_HIT_STREAK_THRESH = 40
REAR_HIT_STREAK_THRESH = 50

# If object is not a threat for consecutive frames, object will still stay in danger status for (FRONT_MAX_STREAK - FRONT_HIT_STREAK_THRESH) / (MISS_MULTIPLIER) frames
FRONT_MAX_STREAK = 80
REAR_MAX_STREAK = 80


# If HIT_mulitplier > MISS_multiplier, a lot of misses will be required to 'deactivate' the Danger status of object.
# number of consecutive frames of hits to trigger danger status = FRONT(REAR)_HIT_STREAK_THRESH / HIT_MULTIPLIER
HIT_MULTIPLIER = 5
# number of consecutive misses to downgrade danger status = FRONT(REAR)_MAX_STREAK / MISS_MULTIPLIER
MISS_MULTIPLIER = 1

# KalmanPointObject (KPO) Parameters

KPO_X_VAR = 1e2
KPO_X_DOT_VAR = 1e1
KPO_Y_VAR = 1e2
KPO_Y_DOT_VAR = 1e1

# Simple Online Realtime Tracker (SORT) parameters

# Kalman filter initialization parameters

# variance of measurement error (set to high if object detector isn't good)
SORT_R_MULT = 1e2
SORT_P1_MULT = 1e2  # variance of state. Is set to high for unknown initial velocities
SORT_P2_MULT = 1e2  # variance of state (besides initial velocites)
SORT_Q_MULT = 0.1  # process noise

# Tracker parameters

SORT_MAX_AGE = 1000
SORT_MIN_HITS = 2
SORT_IOU_THRESH = 0.2

# Audio parameters
FRONT_MIN_RING_GAP = 1  # avoid overlapping audio alarm
REAR_MIN_RING_GAP = 1


# YOLOX Inference Predictor Parameters

YOLOX_CONF = 0.3
YOLOX_NMS = 0.3
YOLOX_TSIZE = None