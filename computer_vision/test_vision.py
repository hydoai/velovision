import sys
sys.path.append('../')

from testing_utils import fuzzy_compare
from vision import core
from argparse import Namespace
from yolox.exp import get_exp


def test_core_cpu():
    args = Namespace(
        experiment_name = None,
        name = None,
        cam0_index = None,
        cam1_index = None,
        vid0_file = "../testing_resources/example_videos/single_overtake.mp4",
        vid1_file = "../testing_resources/example_videos/single_overtaken.mp4",
        cam_type = "GoProHD",
        exp_file = "yolox_exps/nano-foxtrot.py",
        ckpt = None,
        device = "cpu",
        fp16 = True,
        fuse = False,
        trt = False,
        save_result = False,
        view_result = False,
        window_size = 1000,
        tracker_type = "iou",
        view_intercept_result = False,
        save_intercept_result = False,
        production_hardware = False,
        physical_switches = False,
        no_audio = True,
    )
    exp = get_exp(args.exp_file, exp_name=None)

    expected = {
        'num_front_warnings': 1,
        'num_rear_warnings': 1,
        'wall_time_elapsed': 10,
    }
    tolerance = {
        'num_front_warnings': (0,0), # true value minus first value, true value minus second value are all allowed
        'num_rear_warnings': (0,0),
        'wall_time_elapsed': (5,60) # github actions takes ~35 seconds, compared to ~10 seconds on my machine. 
    }
    reality = core(exp,args)
    print(reality)
    
    assert fuzzy_compare(expected, tolerance, reality)

def test_dummy():
    print("Testing a dummy function")
    assert True