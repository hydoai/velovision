from enum import Enum
import math
from math import pi

import numpy as np

import sys
sys.path.append('../')
from utils import CameraFacing, FrameHalf

class Perspective:
    def __init__(self, frame_width=960, cam_fov_deg=45):
        self.frame_width = 960
        self.cam_fov = cam_fov_deg * pi / 180

    def step(self, input_bboxes):
        '''
        Calculate top-view perspective x,y coordinates for each detection

        Expected input:
        np.array of shape (n,10) where
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

        Output:

        an np.array of shape (n,12) where
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
        perspective_output = np.empty((input_bboxes.shape[0], 12))
        perspective_output[:,:10] = input_bboxes[:,:10]

        clean_bboxes = self.preprocess(input_bboxes)
        for i,box in enumerate(clean_bboxes):
            # box = np.array([x1,x2,empty,empty])
            x_coord, y_coord = convert_pov_to_cartesian(box[0], box[1], box[2], self.frame_width, self.cam_fov)
            clean_bboxes[i][2] = x_coord
            clean_bboxes[i][3] = y_coord

        perspective_output[:,(10,11)] = clean_bboxes[:,(2,3)]

        return perspective_output

    def preprocess(self, bboxes):
        '''
        Sanitize some inputs; incoming array has a lot of unneeded information

        Expected input:
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

        Output:

        an np.array of shape (n,5) where
            0 : x_center
            1 : distance
            2 : assigned to front (0) or rear (1)
            3 : empty ( for x coordinate )
            4 : empty ( for y coordinate )
        '''
        clean_bboxes = np.zeros((bboxes.shape[0], 5))
        clean_bboxes[:,(0,1,2)] = bboxes[:,(5,9,7)]
        return clean_bboxes

def convert_pov_to_cartesian(
        x_c,
        distance,
        cam_direction, # 0 : front, 1: rear
        frame_width,
        cam_fov, # must be in radians
        ):
    '''
    Known variables:
        + bounding box center x
        + frame width (pixels)
        + camera field of view

    Output:
        + obj coordinates in x,y with respect to camera as viewed from top

    Conventions:
        + Front: positive x
        + Rear: negative x
        + Left: positive y
        + Right: negative y
    '''
    assert cam_fov <= 2*pi, "Expected camera field of view in RADIANS, less than or equal to 2*pi"

    obj_pix_deviation = abs(x_c - frame_width/2) # pixel deviation from center line

    th_deviation = cam_fov * obj_pix_deviation / frame_width # angle deviation from center line

    cam_facing = CameraFacing.FRONT if cam_direction == 0 else CameraFacing.REAR

    frame_half = FrameHalf.LEFT if x_c < frame_width/2 else FrameHalf.RIGHT

    if cam_facing == CameraFacing.FRONT:
        if frame_half == FrameHalf.LEFT:
            th_abs = th_deviation
        if frame_half == FrameHalf.RIGHT:
            th_abs= 2*pi - th_deviation
        else:
            assert False

    if cam_facing == CameraFacing.REAR:
        if frame_half == FrameHalf.LEFT:
            th_abs= pi + th_deviation
        if frame_half == FrameHalf.RIGHT:
            th_abs= pi - th_deviation
        else:
            assert False

    x_coord = distance * math.cos(th_abs)
    y_coord = distance * math.sin(th_abs)

    return x_coord, y_coord


