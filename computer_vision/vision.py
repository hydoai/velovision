#!/usr/bin/python3
import argparse
import sys
sys.path.append('../')
import os
import time
from loguru import logger
import copy
import cv2
import numpy as np
import torch
from multiprocessing import Process

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from yolox.utils import vis

from PARAMETERS import *

from subvision.yolox_inference.predictor import Predictor
from subvision.sort.sort_minimal import Sort
from subvision.iou_tracker.iou_tracker import IOUTracker # faster but worse than SORT
from subvision.watchout.watchout import Watchout
from subvision.watchout.custom_vis import distance_custom_vis
from subvision.perspective.perspective import Perspective
from subvision.intercept.intercept import Intercept
from subvision.intercept.custom_vis import TTE_EP_custom_vis, concat_plot_to_frame
from subvision.intercept.pyplot_vis import plot_birds_eye
from subvision.utils import center_crop, combine_dets, split_dets

from debug_utils.avgtimer import AvgTimer # timer with rolling averages

from sensing_interface.cameras import CameraInterface

from feedback_interface.sounds import GiSpeaker

def make_parser():
    parser = argparse.ArgumentParser("YOLOX for Hydo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name (select from pretrained)")
    parser.add_argument("-cam0", "--cam0_index", type=int, default=None, help="Camera 0 index. See 'ls /dev/video*'")
    parser.add_argument("-cam1", "--cam1_index", type=int, default=None, help="Camera 1 index. See 'ls /dev/video*'")
    parser.add_argument("-vid0", "--vid0_file", type=str, default=None, help="Video 0 file path.")
    parser.add_argument("-vid1", "--vid1_file", type=str, default=None, help="Video 1 file path.")
    parser.add_argument("--crop0_width", type=int, default=960, help="Width of center-cropped video 0 ")
    parser.add_argument("--crop0_height", type=int, default=540, help="Height of center-cropped video 0")
    parser.add_argument("--crop1_width", type=int, default=960, help="Width of center-cropped video 1")
    parser.add_argument("--crop1_height", type=int, default=540, help="height of center-cropped video 1")
    parser.add_argument("-f", "--exp_file", type=str, default='yolox_exps/nx-alpha.py', help="Please input your experiment description python file (in 'exps' folder)")
    parser.add_argument("-c", "--ckpt", type=str, default=None, help="Specify specific ckpt for evaluation. Otherwise, best_ckpt of exp_file will be used")
    parser.add_argument("--device", type=str, default='gpu', help="Specify device to run model. 'cpu' or 'gpu'")
    parser.add_argument("--conf", type=float, default=0.3, help="Test object confidence threshold")
    parser.add_argument("--nms", type=float, default=0.3, help="Test non max suppression threshold")
    parser.add_argument("--tsize", type=int, default=None, help="Test image size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopt mixed precision evaluation")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest='trt', default=False, action="store_true", help="Use TensorRT for faster inference. Always use on Jetson")
    parser.add_argument("--save_result", action="store_true", help="whether to save inference result")
    parser.add_argument("--view_result", action="store_true", help="whether to view inference result live (slow)")
    parser.add_argument("--window_size", type=int, default=1000, help="if --view_result is True, set this window size. Larger makes it much slower.")
    parser.add_argument("--debug", action='store_true', help="Non-essential features for development. If --view_result or --save_result is set, then this will automatically be set true.")

    parser.add_argument("--tracker_type", type=str, default='iou', help="Choose betwen 'iou' and 'sort' trackers. IOU Tracker is faster, SORT is smoother and better.")
    parser.add_argument("--view_intercept", action='store_true', help="View live matplotlib visualization of birds-eye view. This automatically eables '--view_result' option")
    parser.add_argument("--save_intercept", action='store_true', help="Save every frame of matplotlib visualization to 'subvision/intercept/saved_figs'")

    # JETSON PROTOTYPE HARDWARE SPECIFIC SETTINGS
    parser.add_argument("--production_hardware", default=False, action="store_true", help="Production hardware specific camera setups and other settings")
    parser.add_argument("--physical_switches", default=False, action="store_true", help="GPIO hardware controls")
    return parser

def main(exp, args):

    if args.physical_switches:
        from ui_input.physical_input import Pins, safe_shutdown
        pins = Pins()
        pins.setup_function('shutdown', safe_shutdown)
        pins.start()


    def start_camera_pipelines():
        camera_interface = CameraInterface()
        camera_interface.start_pipelines()
        camera_interface.record_nvenc_h265(4, 1280, 720, fps=30, max_length=20)
        camera_interface.record_nvenc_h265(6, 640, 480, fps=30, max_length=20)

    if args.production_hardware:
        p = Process(target=start_camera_pipelines)
        p.start()
        time.sleep(10)

    if args.save_result or args.view_result or args.view_intercept:
        args.debug = True
    if args.view_intercept:
        args.view_result = True
    if args.view_intercept and args.save_result:
        logger.error("Saving the concatenated two vides + intercept plot is not implemented, because VideoWriter needs to know the final output shape. It's possible, but I haven't bothered yet.\n In the meantime, use '--view_result' instead of '--save_result'")
        raise NotImplementedError

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, 'visualized_results')
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info(f"Args: {args}")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info(f"Model summary: {get_model_info(model, exp.test_size)}")

    if args.device=='gpu':
        model.cuda()
        if args.fp16:
            model.half() # convert model to fp16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info('Loading checkpoint...')
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        logger.info('Done loading checkpoint!')

    if args.fuse:
        logger.info('Fusing model...')
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model does not support fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(trt_file), "TensorRT model not found! Please run 'python3 tools/trt.py' to create a TRT model first"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT for inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, HYDO_CLASSES, trt_file, decoder, args.device, args.fp16)

    # CAPTURE FRAMES FROM VIDEO STREAM

    if args.production_hardware:
        args.cam0_index = 3
        args.cam1_index = 5

    cap0 = cv2.VideoCapture(args.vid0_file if args.cam0_index is None else args.cam0_index)
    cap1 = cv2.VideoCapture(args.vid1_file if args.cam1_index is None else args.cam1_index)

    width0 = cap0.get(cv2.CAP_PROP_FRAME_WIDTH)
    width1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    if width0 != width1:
        logger.warning("Widths do not match. Using larger of the two.")
        logger.warning(f"Width of cap0: {width0}")
        logger.warning(f"Width of cap1: {width1}")
    width = max(width0, width1)

    height0 = cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if height0 != height1:
        logger.warning("Heights do not match.")
        logger.warning(f"Height of cap0: {height0}")
        logger.warning(f"Height of cap1: {height1}")
    height = height0 + height1

    fps0 = cap0.get(cv2.CAP_PROP_FPS)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    if fps0 != fps1:
        logger.warning("FPS does not match. Using lower of the two FPSs")
        logger.warning(f"FPS of cap0: {fps0}")
        logger.warning(f"FPS of cap1: {fps1}")
    fps = min(fps0, fps1)


    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, "output.mp4")

        logger.info(f"Video output is saved to {save_path}")

        if args.crop0_width is not None:
            width = max(args.crop0_width, args.crop1_width)
            height = args.crop0_height + args.crop1_height


        vid_writer = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))


    avgtimer = AvgTimer()
    sort_tracker = Sort(max_age = SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESH)
    iou_tracker = IOUTracker()
    watchout_f = Watchout()
    watchout_r = Watchout()
    perspective = Perspective()
    intercept = Intercept(view_vis=args.view_intercept, save_vis=args.save_intercept) # matplotlib visualizations saved to subvison/intercept/saved_figs
    gi_speaker = GiSpeaker()

    while cap0.isOpened() and cap1.isOpened():
        ret_val0, frame0 = cap0.read()
        ret_val1, frame1 = cap1.read()
        if args.production_hardware:
            resize_width = 1280
            resize_height = 720
            frame1 = cv2.resize(frame1, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        if not (ret_val0 and ret_val1):
            break
        else:
            avgtimer.start('frame')
            avgtimer.start('center_crop')

            frame0 = center_crop(frame0, args.crop0_width, args.crop0_height, nudge_down=FRONT_NUDGE_DOWN, nudge_right=FRONT_NUDGE_RIGHT)
            frame1 = center_crop(frame1, args.crop1_width, args.crop1_height, nudge_down=REAR_NUDGE_DOWN, nudge_right=REAR_NUDGE_RIGHT)
            #import IPython; IPython.embed()
            avgtimer.end('center_crop')

            avgtimer.start('predictor')
            outputs, img_info = predictor.inference(frame0, frame1)
            avgtimer.end('predictor')

            avgtimer.start('expanded_outputs')
            expanded_outputs = predictor.expand_bboxes(outputs, img_info)
            avgtimer.end('expanded_outputs')

            detections = expanded_outputs

            front_watchout_output = None
            rear_watchout_output = None
            if detections is not None:
                avgtimer.start('tracker')

                # convert numpy array into list of dictionaries
                if args.tracker_type == 'iou':
                    det_list = []
                    for det in detections:
                        det_list.append({
                            'bbox': (det[0], det[1], det[2], det[3]),
                            'score': det[4],
                            'class': det[5],})
                    iou_output = iou_tracker.update(det_list)

                    # convert list of dictionaries back into numpy array
                    det_array = np.zeros((len(iou_output),9))

                    for i,track in enumerate(iou_output):
                        det_array[i][0:4] = iou_output[i]['bboxes']
                        det_array[i][4] = iou_output[i]['classes']
                        det_array[i][8] = iou_output[i]['tracking_id']
                    tracked_output = det_array


                if args.tracker_type == 'sort':
                    tracked_output = sort_tracker.update(detections)
                avgtimer.end('tracker')

                # split detections into front and rear, then run watchout separately
                avgtimer.start('split_dets')

                front_dets, rear_dets = split_dets(tracked_output, args.crop0_width, args.crop0_height)
               #
                avgtimer.end('split_dets')
                avgtimer.start('watchout')
                front_watchout_output = watchout_f.step(front_dets)
                rear_watchout_output = watchout_r.step(rear_dets)
                avgtimer.end('watchout')

                # then combine the split detections back into one array for visualization.
                avgtimer.start('combine_dets')
                combined_dets = combine_dets(front_watchout_output, rear_watchout_output, args.crop0_height)
                avgtimer.end('combine_dets')


            img = img_info['raw_img']
            if outputs[0] is None:
                result_frame = img
            else:
                #import IPython; IPython.embed()
                output = outputs[0].cpu()
                bboxes = combined_dets[:,0:4]
                cls = combined_dets[:,4]
                scores = output[:,4] * output[:,5]
                cls_conf = 0.35
                distance = combined_dets[:,9]
                track_id = combined_dets[:,8]

                #result_frame = vis(img,bboxes,scores,cls,cls_conf, predictor.cls_names)
                #result_frame = distance_custom_vis(img,bboxes,scores,cls, distance, track_id, cls_conf, predictor.cls_names)

                avgtimer.start('perspective')
                perspective_output = perspective.step(combined_dets)
                avgtimer.end('perspective')

                avgtimer.start('intercept')
                front_dangers, rear_dangers, plot_image = intercept.step(perspective_output)

                # optional filtering by y center
                front_dangers, rear_dangers = intercept.curtain_filter(front_dangers, rear_dangers)

                avgtimer.end('intercept')

                if args.debug:
                    result_frame = TTE_EP_custom_vis(img, bboxes, scores, cls, distance, front_dangers, rear_dangers, track_id, conf=0.5, class_names=HYDO_CLASSES)
                    if args.view_intercept:
                        assert plot_image is not None, "Expected intercept.step() to return a matplotlib image as a numpy array"
                        result_frame = concat_plot_to_frame(result_frame, plot_image) # puts 'plot_image' plot underneath the 'result_frame' picture.

                avgtimer.start('intercept_ring')
                front_ring_now, rear_ring_now = intercept.should_ring_now() # ring once when danger starts
                avgtimer.end('intercept_ring')

                if front_ring_now:
                    if args.physical_switches:
                        if pins.bool['front_toggle']:
                            logger.warning("Front warning triggered")
                            gi_speaker.play_right() # front speaker is connected to right channel
                    else:
                        logger.warning("Front warning triggered")
                        gi_speaker.play_right()
                    # FEATURE REQUEST : change volume depending on distance

                if rear_ring_now:
                    if args.physical_switches:
                        if pins.bool['rear_toggle']:
                            logger.warning("Rear warning triggered")
                            gi_speaker.play_left()
                    else:
                        logger.warning("Rear warning triggered")
                        gi_speaker.play_left()



            if args.save_result:
                vid_writer.write(result_frame)
            if args.view_result:
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('result', args.window_size,args.window_size) # runtime speed depends heavily on size
                cv2.imshow("result", result_frame)
                cv2.waitKey(1)
            ch = cv2.waitKey(1)
            if ch==27 or ch==ord('q') or ch==ord('Q'):
                break


            avgtimer.end('frame')
            logger.info('\n')
            logger.info(f"{round(1/(avgtimer.rolling_avg('frame')),2)} FPS")
            logger.info(f"Center crop: {avgtimer.rolling_avg('center_crop')} seconds")
            logger.info(f"Predictor: {avgtimer.rolling_avg('predictor')} seconds")
            logger.info(f"Tracker: {avgtimer.rolling_avg('tracker')} seconds")
            logger.info(f"Split dets: {avgtimer.rolling_avg('split_dets')} seconds")
            logger.info(f"Watchout: {avgtimer.rolling_avg('watchout')} seconds")
            logger.info(f"Combine dets: {avgtimer.rolling_avg('combine_dets')} seconds")
            logger.info(f"Perspective: {avgtimer.rolling_avg('perspective')} seconds")
            logger.info(f"Intercept: {avgtimer.rolling_avg('intercept')} seconds")
            logger.info(f"Intercept Ring: {avgtimer.rolling_avg('intercept_ring')} seconds")
            #logger.info(f"{len(outputs[0]) if outputs[0] is not None else 0} objects detected")

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
    print('end')
