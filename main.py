import argparse
import os
import time
from loguru import logger

import cv2
import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

# Classes
HYDO_CLASSES = ['bicycle', 'bus', 'car', 'cyclist', 'motorcycle', 'pedestrian', 'truck']

# rolling average timer
from avgtimer import AvgTimer

# YOLOX predictor
from yolox_inference import Predictor

# SORT
from gi_sort.sort_minimal import *

# Watchout
from gi_watchout.watchout import *

# Jetson inference/utils lib
import jetson.inference
import jetson.utils

def make_parser():
    parser = argparse.ArgumentParser("HYDO on Jetson Nano")
    parser.add_argument("mode", default="video", help="input type, eg. video and camera")
    parser.add_argument("--path", default="VIDEO_PATH", help="path to input video")
    parser.add_argument("--camid", type=int, default=0, help="camera id")
    parser.add_argument("--save_result", action="store_true", help="save inference result of video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file. See exps folder for examples")
    parser.add_argument("--conf", default=0.3, type=float, help="test confidence")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mixed precision evaluation")
    parser.add_argument("--fuse", dest="fuse", default=False, action = "store_true", help="Fuse conv and bn for testing")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using tensorRT for testing")
    return parser


def imageflow(predictor, vis_folder, current_time, args):

    # profiling
    avg_timer = AvgTimer()
    
    sort_tracker = Sort()

    # watchout parameters
    wo_names = HYDO_CLASSES
    wo_names_to_height_dict = {}
    wo_width = 320 
    wo_height = 320 
    watchout = Watchout(wo_names, wo_names_to_height_dict, wo_width, wo_height)

    cap = cv2.VideoCapture(args.path  if args.mode == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        os.makedirs(save_folder, exist_ok=True)
        if args.mode == "video":
            save_path = os.path.join(save_folder, args.path.split("/")[-1])
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        
        logger.info(f"Video is saved to: {save_path}")

        vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

    rolling_sums = [] # frame time calculation
    while cap.isOpened():
        avg_timer.start("frame")
        
        avg_timer.start("read_video")
        ret_val, frame = cap.read()
        avg_timer.end("read_video")
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret_val:
            avg_timer.start("YOLOX_inference")
            outputs, img_info = predictor.inference(frame)
            avg_timer.end("YOLOX_inference")

            if args.save_result:
                avg_timer.start("YOLOX_visual")
                result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
                avg_timer.end("YOLOX_visual")

            # predictor outputs format: a 2d tensor where each row is:
            # [x1,y1,x2,y2, object-ness confidence, class confidence, predicted class index]

            # SORT expected input format: 2d np array where each row is:
            # [x1,y1,x2,y2,class confidence, predicted class index]

            # Convert Predictor outputs to SORT input

            if outputs[0] is not None:
                avg_timer.start("SORT_preprocess")
                pred_box = outputs[0][:,:4].cpu().numpy()
                class_conf = outputs[0][:,5:6].cpu().numpy()
                class_ind = outputs[0][:,6:7].cpu().numpy()
                track_input = np.hstack((pred_box, class_conf, class_ind))
                avg_timer.end("SORT_preprocess")
                avg_timer.start("SORT")
                track_output = sort_tracker.update(track_input)
                avg_timer.end("SORT")
                avg_timer.start("watchout")
                watchout_output = watchout.step(track_output)
                avg_timer.end("watchout")

            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        avg_timer.end("frame")
        logger.info(f"Rolling frame rate: {1/avg_timer.rolling_avg('frame')} FPS")
        logger.info(f"Read video: {avg_timer.rolling_avg('read_video')}")
        logger.info(f"YOLOX inference: {avg_timer.rolling_avg('YOLOX_inference')}")
        #logger.info(f"YOLOX visual: {avg_timer.rolling_avg('YOLOX_visual')}")

        logger.info(f"SORT preprocess: {avg_timer.rolling_avg('SORT_preprocess')}")
        logger.info(f"SORT: {avg_timer.rolling_avg('SORT')}")
        logger.info(f"Watchout: {avg_timer.rolling_avg('watchout')}")
        print('\n')
def main(exp, args):
    args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device= 'gpu'

    logger.info(f"Args: {args}")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half() # convert model to fp16
    model.eval()

    if args.fuse:
        logger.info("\t Fusing model..")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model does not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
                trt_file), "TensorRT model is not found!\n Run `python3 tools/trt.py` first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT for inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, HYDO_CLASSES, trt_file, decoder, args.device, args.fp16)

    current_time = time.localtime()

    if args.mode == "video" or args.mode == "camera":
        imageflow(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    args.name = None
    exp = get_exp(args.exp_file, args.name)

    main(exp,args)







