#!/usr/bin/python3
import sys
sys.path.append('../')
import argparse
import os
import time
from loguru import logger
import cv2
import numpy as np
import torch
from multiprocessing import Process
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from PARAMETERS import *
from subvision.yolox_inference.predictor import Predictor
from subvision.sort.sort_minimal import Sort
from subvision.iou_tracker.iou_tracker import IOUTracker  # faster but worse than SORT
from subvision.watchout.watchout import Watchout
from subvision.watchout.custom_vis import distance_custom_vis
from subvision.perspective.perspective import Perspective
from subvision.intercept.intercept import Intercept
from subvision.intercept.custom_vis import TTE_EP_custom_vis, concat_plot_to_frame
from subvision.intercept.pyplot_vis import plot_birds_eye
from subvision.utils import fixed_aspect_ratio_crop, combine_dets, split_dets
from debug_utils.avgtimer import AvgTimer  # timer with rolling averages
from sensing_interface.cameras import CameraInterface
from feedback_interface.sounds import GiSpeaker
from sdcard_management.free_space import threaded_free_space
from sdcard_management.remount_sd_card import remount_sd_card, get_username, device_path, mount_path

# LeetopA203 specific: microSD card slot is at /dev/mmcblk1p1; modify this at remound_sdcard.py

logger.remove()
logger.add(sys.stderr, level="WARNING")
# INFO for profiling
# WARNING for warning alarm triggers

def start_camera_pipelines():
    camera_interface = CameraInterface()
    camera_interface.start_pipelines()
    camera_interface.record_nvenc_h265(4, 1280, 720, fps=30, max_length=20)
    camera_interface.record_nvenc_h265(6, 640, 480, fps=30, max_length=20)

def make_parser():
    parser = argparse.ArgumentParser("YOLOX for Hydo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="model name (select from pretrained)")
    parser.add_argument("-cam0", "--cam0_index", type=int,
                        default=None, help="Camera 0 index. See 'ls /dev/video*'")
    parser.add_argument("-cam1", "--cam1_index", type=int,
                        default=None, help="Camera 1 index. See 'ls /dev/video*'")
    parser.add_argument("-vid0", "--vid0_file", type=str,
                        default=None, help="Video 0 file path.")
    parser.add_argument("-vid1", "--vid1_file", type=str,
                        default=None, help="Video 1 file path.")

    parser.add_argument("--cam_type", type=str, default="GoProHD", 
                        help="Camera type. Choose among 'GoProHD' and 'DK1'. See PARAMETERS.py for details. This sets the 'caminfo0' and 'caminfo1' internal variables.")

    parser.add_argument("-f", "--exp_file", type=str, default='yolox_exps/nx-foxtrot.py',
                        help="Please input your experiment description python file (in 'exps' folder)")
    parser.add_argument("-c", "--ckpt", type=str, default=None,
                        help="Specify specific ckpt for evaluation. Otherwise, best_ckpt of exp_file will be used")
    parser.add_argument("--device", type=str, default='gpu',
                        help="Specify device to run model. 'cpu' or 'gpu'")
    parser.add_argument("--fp16", dest="fp16", default=True,
                        action="store_true", help="Adopt mixed precision evaluation")
    parser.add_argument("--fuse", dest="fuse", default=False,
                        action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest='trt', default=False, action="store_true",
                        help="Use TensorRT for faster inference. Always use on Jetson")
    parser.add_argument("--save_result", action="store_true",
                        help="whether to save inference result")
    parser.add_argument("--view_result", action="store_true",
                        help="whether to view inference result live (slow)")
    parser.add_argument("--window_size", type=int, default=1000,
                        help="if --view_result is True, set this window size. Larger makes it much slower.")

    parser.add_argument("--tracker_type", type=str, default='iou',
                        help="Choose betwen 'iou' and 'sort' trackers. IOU Tracker is faster, SORT is smoother and better.")
    parser.add_argument("--view_intercept_result", action='store_true',
                        help="View live matplotlib visualization of birds-eye view. This automatically eables '--view_result' option")
    parser.add_argument("--save_intercept_result", action='store_true',
                        help="Save every frame of matplotlib visualization to 'subvision/intercept/saved_figs'")

    # JETSON PROTOTYPE HARDWARE SPECIFIC SETTINGS
    parser.add_argument("--production_hardware", default=False, action="store_true",
                        help="Shortcut to override some arguments with HYDO DevKit-One specific camera, storage, and other configurations")
    parser.add_argument("--physical_switches", default=False,
                        action="store_true", help="GPIO hardware controls")
    parser.add_argument("--no_audio", default=False,
                        action="store_true", help="Disable pygame ALSA audio output for Docker-based headless testing (github actions)")
    return parser

def resolve_args(args):
    '''
    Resolve dependencies between arguments
    Configure correct args for jetson and desktop runs

    Only parse between arguments, no external variables or side effects!
    '''

    if args.production_hardware:
        args.cam_type = 'DK1'
        args.cam0_index = 3
        args.cam1_index = 5
        args.vid0_file = '/dev/video0'
        args.vid1_file = '/dev/video1'
        args.trt = True
        args.device = 'gpu'
        args.save_result = False
        args.view_result = False
        args.tracker_type = 'iou'
        args.view_intercept_result = False 
        args.save_intercept_result = False
        args.no_audio = False
        args.physical_switches = True

    if args.trt:
        args.device = 'gpu'

    if args.view_intercept_result and args.save_result:
        logger.error("Saving the concatenated two vides + intercept plot is not implemented, because VideoWriter needs to know the final output shape. It's possible, but I haven't bothered yet.\n In the meantime, use '--view_result' instead of '--save_result'")
        raise NotImplementedError
    
    print("args.device:")
    print(args.device)
    return args

def core(exp, args):
    main_results = { # counter for testing
        'num_front_warnings': 0,
        'num_rear_warnings': 0,
        'wall_time_elapsed': time.time(),
    }

    if args.cam_type == "GoProHD":
        caminfo0 = GoProHD_front
        caminfo1 = GoProHD_rear
    elif args.cam_type == "DK1":
        caminfo0 = DK1_frontcam
        caminfo1 = DK1_rearcam

    if args.production_hardware:
        # remounting to ensuring that the inserted microSD card is at /media/[username]/microsdcard
        remount_sd_card(device_path, mount_path)
        time.sleep(2)
        # run thread to occasionally delete old videos on microsdcard
        threaded_free_space(mount_path, min_space_remaining=10)

    if args.physical_switches:
        from ui_input.physical_input import Pins, safe_shutdown
        pins = Pins()
        pins.setup_function('shutdown', safe_shutdown)
        pins.start()

    if args.production_hardware:
        p = Process(target=start_camera_pipelines)
        p.start()
        time.sleep(10)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, 'visualized_results')
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = None

    logger.info(f"Args: {args}")

    if YOLOX_CONF is not None:
        exp.test_conf = YOLOX_CONF
    if YOLOX_NMS is not None:
        exp.nmsthre = YOLOX_NMS
    if YOLOX_TSIZE is not None:
        exp.test_size = (YOLOX_TSIZE, YOLOX_TSIZE)

    model = exp.get_model()
    logger.info(f"Model summary: {get_model_info(model, exp.test_size)}")

    if args.device == 'gpu':
        model.cuda()
        if args.fp16:
            model.half()  # convert model to fp16
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
        assert os.path.exists(
            trt_file), "TensorRT model not found! Please run 'python3 tools/trt.py' to create a TRT model first"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT for inference")
    else:
        trt_file = None
        decoder = None


    cap0 = cv2.VideoCapture(
        args.vid0_file if args.cam0_index is None else args.cam0_index)
    cap1 = cv2.VideoCapture(
        args.vid1_file if args.cam1_index is None else args.cam1_index)

    cap0_width = cap0.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap0_height = cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap1_width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap1_height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # sanity check: CameraInfo width & height should be same as the cap width & height.
    # if fail, check CameraInfo setup in PARAMETERS.py
    try: 
        assert caminfo0.width_res == cap0_width
        assert caminfo0.height_res == cap0_height
        assert caminfo1.width_res == cap1_width
        assert caminfo1.height_res == cap1_height
    except AssertionError:
        print("ERROR: Cameras not set up properly.")
        print("cap0_height:", cap0_height)
        print("cap0_width:", cap0_width)
        print("caminfo0.height_res:", caminfo0.height_res)
        print("caminfo0.width_res:", caminfo0.width_res)
        print("cap1_height:", cap1_height)
        print("cap1_width:", cap1_width)
        print("caminfo1.height_res:", caminfo1.height_res)
        print("caminfo1.width_res:", caminfo1.width_res)


    fps0 = cap0.get(cv2.CAP_PROP_FPS)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps = max(fps0, fps1)
    # In inference loop,
    # crop input frame to CameraInfo.inference_crop_ratio
    # then resize to unified size

    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime(
            "%Y_%m_%d_%H_%M_%S", time.localtime()))
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, "output.mp4")

        logger.info(f"Video output is saved to {save_path}")

        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(
            *"mp4v"), fps, (int(UNIFIED_WIDTH), int(UNIFIED_HEIGHT)))

    avgtimer = AvgTimer()

    if args.tracker_type == 'sort':
        sort_tracker = Sort(
            max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESH)
    elif args.tracker_type == 'iou':
        iou_tracker = IOUTracker()
    else:
        logging.error("Please choose valid tracker type among 'sort', 'iou'")
        raise NotImplementedError

    predictor = Predictor(model, exp, HYDO_CLASSES,
                          trt_file, decoder, args.device, args.fp16)
    watchout_f = Watchout(camera_info=caminfo0)
    watchout_r = Watchout(camera_info=caminfo1)
    perspective = Perspective(frame_width= UNIFIED_WIDTH, cam_fov_deg_f=caminfo0.width_fov, cam_fov_deg_r=caminfo1.width_fov)

    # matplotlib visualizations saved to subvison/intercept/saved_figs
    intercept = Intercept(view_vis=args.view_intercept_result,
                          save_vis=args.save_intercept_result,
                          frame_width=UNIFIED_WIDTH)
    if not args.no_audio:
        gi_speaker = GiSpeaker()
    else:
        gi_speaker = None

    if cap0.isOpened() and cap1.isOpened() and not args.no_audio:
        gi_speaker.play_startup

    while cap0.isOpened() and cap1.isOpened():
        ret_val0, frame0 = cap0.read()
        ret_val1, frame1 = cap1.read()
        if not (ret_val0 and ret_val1):
            break 
        else:
            avgtimer.start('frame')

            avgtimer.start('pre_crop')
            frame0 = fixed_aspect_ratio_crop(
                frame0,
                caminfo0,
                to_width=UNIFIED_WIDTH,
                to_height=UNIFIED_HEIGHT)
            frame1 = fixed_aspect_ratio_crop(
                frame1,
                caminfo1,
                to_width=UNIFIED_WIDTH,
                to_height=UNIFIED_HEIGHT)
            avgtimer.end('pre_crop')

            avgtimer.start('predictor')
            predictor_outputs, img_info = predictor.inference(frame0, frame1)
            avgtimer.end('predictor')

            avgtimer.start('expanded_outputs')
            detections = predictor.expand_bboxes(predictor_outputs, img_info)
            avgtimer.end('expanded_outputs')

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
                            'class': det[5], })
                    iou_output = iou_tracker.update(det_list)

                    # convert list of dictionaries back into numpy array
                    det_array = np.zeros((len(iou_output), 9))

                    for i, track in enumerate(iou_output):
                        det_array[i][0:4] = iou_output[i]['bboxes']
                        det_array[i][4] = iou_output[i]['classes']
                        det_array[i][8] = iou_output[i]['tracking_id']
                    tracked_output = det_array

                if args.tracker_type == 'sort':
                    tracked_output = sort_tracker.update(detections)
                avgtimer.end('tracker')

                # split detections into front and rear, then run watchout separately
                avgtimer.start('split_dets')
                front_dets, rear_dets = split_dets(
                    tracked_output, UNIFIED_HEIGHT)
                avgtimer.end('split_dets')

                avgtimer.start('watchout')
                front_watchout_output = watchout_f.step(front_dets)
                rear_watchout_output = watchout_r.step(rear_dets)
                avgtimer.end('watchout')

                # then combine the split detections back into one array for visualization.
                avgtimer.start('combine_dets')
                combined_dets = combine_dets(
                    front_watchout_output, rear_watchout_output, UNIFIED_HEIGHT)
                avgtimer.end('combine_dets')
            else:
                pass  # no detections

            img = img_info['raw_img']
            if predictor_outputs[0] is None:
                result_frame = img
            else:
                output = predictor_outputs[0].cpu()
                bboxes = combined_dets[:, 0:4]
                cls = combined_dets[:, 4]
                scores = output[:, 4] * output[:, 5]
                distance = combined_dets[:, 9]
                track_id = combined_dets[:, 8]

                # legacy visualizations
                #result_frame = vis(img,bboxes,scores,cls,cls_conf, predictor.cls_names)
                #result_frame = distance_custom_vis(img,bboxes,scores,cls, distance, track_id, cls_conf, predictor.cls_names)

                avgtimer.start('perspective')
                perspective_output = perspective.step(combined_dets)
                avgtimer.end('perspective')

                avgtimer.start('intercept')
                front_dangers, rear_dangers, plot_image = intercept.step(
                    perspective_output)

                # optional simple filtering by y center
                front_dangers, rear_dangers = intercept.curtain_filter(
                    front_dangers, rear_dangers)

                avgtimer.end('intercept')

                if args.save_result or args.view_result or args.view_intercept_result:
                    result_frame = TTE_EP_custom_vis(
                        img, bboxes, scores, cls, distance, front_dangers, rear_dangers, track_id, conf=0.5, class_names=HYDO_CLASSES)
                    if args.view_intercept_result:
                        assert plot_image is not None, "Expected intercept.step() to return a matplotlib image as a numpy array"
                        # puts 'plot_image' plot underneath the 'result_frame' picture.
                        result_frame = concat_plot_to_frame(
                            result_frame, plot_image)

                avgtimer.start('intercept_ring')
                # ring once when danger starts
                front_ring_now, rear_ring_now = intercept.should_ring_now()
                avgtimer.end('intercept_ring')

                play_front_sound = False
                play_rear_sound = False
                if front_ring_now:
                    if args.physical_switches:
                        if pins.bool['front_toggle']:
                            play_front_sound = True
                    else:
                        play_front_sound = True

                if rear_ring_now:
                    if args.physical_switches:
                        if pins.bool['rear_toggle']:
                            play_rear_sound = True
                    else:
                        play_rear_sound = True

                if play_front_sound:
                    logger.warning("Front warning triggered")
                    main_results['num_front_warnings'] += 1
                    if not args.no_audio:
                        gi_speaker.play_right()

                if play_rear_sound:
                    logger.warning("Rear warning triggered")
                    main_results['num_rear_warnings'] += 1
                    if not args.no_audio:
                        gi_speaker.play_left()

            if args.save_result:
                vid_writer.write(result_frame)
            if args.view_result:
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                # runtime speed depends heavily on size
                cv2.resizeWindow('result', args.window_size, args.window_size)
                cv2.imshow("result", result_frame)
                cv2.waitKey(1)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break

            avgtimer.end('frame')
            logger.info('\n')
            logger.info(f"{round(1/(avgtimer.rolling_avg('frame')),2)} FPS")
            logger.info(
                f"Center crop: {avgtimer.rolling_avg('pre_crop')} seconds")
            logger.info(
                f"Predictor: {avgtimer.rolling_avg('predictor')} seconds")
            logger.info(f"Tracker: {avgtimer.rolling_avg('tracker')} seconds")
            logger.info(
                f"Split dets: {avgtimer.rolling_avg('split_dets')} seconds")
            logger.info(
                f"Watchout: {avgtimer.rolling_avg('watchout')} seconds")
            logger.info(
                f"Combine dets: {avgtimer.rolling_avg('combine_dets')} seconds")
            logger.info(
                f"Perspective: {avgtimer.rolling_avg('perspective')} seconds")
            logger.info(
                f"Intercept: {avgtimer.rolling_avg('intercept')} seconds")
            logger.info(
                f"Intercept Ring: {avgtimer.rolling_avg('intercept_ring')} seconds")
            #logger.info(f"{len(outputs[0]) if outputs[0] is not None else 0} objects detected")

    main_results['wall_time_elapsed'] = time.time() - main_results['wall_time_elapsed']
    return main_results


if __name__ == "__main__":
    args = make_parser().parse_args()
    resolved_args = resolve_args(args)
    exp = get_exp(resolved_args.exp_file, args.name)
    main_results = core(exp, resolved_args)
    print("finished")
