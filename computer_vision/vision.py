import argparse
import os
import time
from loguru import logger
import copy
import cv2
import numpy as np
import torch

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
#from yolox.utils.visualize import _COLORS

from subvision.sort.sort_minimal import Sort
from subvision.watchout.watchout import Watchout
from subvision.perspective.perspective import Perspective
from subvision.utils import center_crop, combine_dets, split_dets
from insight.overlayed_vis import custom_vis

from debug_utils.avgtimer import AvgTimer # timer with rolling averages


HYDO_CLASSES = ['bicycle', 'bus', 'car', 'cyclist', 'motorcycle', 'pedestrian', 'truck'] # alphabetically ordered because that's how it's trained


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
    parser.add_argument("-f", "--exp_file", type=str, default=None, help="Please input your experiment description python file (in 'exps' folder)")
    parser.add_argument("-c", "--ckpt", type=str, default=None, help="Specify specific ckpt for evaluation. Otherwise, best_ckpt of exp_file will be used")
    parser.add_argument("--device", type=str, default='gpu', help="Specify device to run model. 'cpu' or 'gpu'")
    parser.add_argument("--conf", type=float, default=0.3, help="Test object confidence threshold")
    parser.add_argument("--nms", type=float, default=0.3, help="Test non max suppression threshold")
    parser.add_argument("--tsize", type=int, default=None, help="Test image size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopt mixed precision evaluation")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest='trt', default=False, action="store_true", help="Use TensorRT for faster inference. Always use on Jetson")
    parser.add_argument("--save_result", action="store_true", help="whether to save inference result")
    return parser

class Predictor:
    def __init__(
            self,
            model,
            exp,
            cls_names = HYDO_CLASSES,
            trt_file = None,
            decoder = None,
            device = 'cpu',
            fp16 = False,
            ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform()
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1,3,exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x) # run once
            self.model = model_trt

    def inference(self, img0, img1=None):
        '''
        Arguments
            img0: input image for front camera
            img1: input image for rear camera
        '''
        img_info = {"id": 0}

        if img1 is None:
            height, width = img0.shape[:2]
            raw_img = img0
            ratio = min(self.test_size[0] / img0.shape[0], self.test_size[1] / img0.shape[1])
        else:
            height = img0.shape[0] + img1.shape[0]
            width = max(img0.shape[1], img1.shape[1])
            raw_img = np.append(img0, img1, axis=0)
            ratio = min(self.test_size[0] / raw_img.shape[0], self.test_size[1] / raw_img.shape[1])

        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = raw_img
        img_info['ratio'] = ratio

        # preprocessing
        img, _ = self.preproc(raw_img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0) # adds one dimension to front. e.g. (3,640,640) to (1,3,640,640)
        img = img.float() # uint8 -> float32

        if self.device == 'gpu':
            img = img.cuda()
            if self.fp16:
                img = img.half() # to fp16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)
        return outputs, img_info

    def expand_bboxes(self, output, img_info):
        '''
        move neural network output to CPU,
        expand bboxes to raw_img's scale

        '''
        ratio = img_info['ratio']
        if output[0] is None:
            return None
        output = output[0].cpu().numpy()
        bboxes = output[:,0:4]

        # resize bboxes from neural network's output size to raw_img's size
        bboxes /= ratio
        cls = output[:,6:7]
        scores = output[:,4:5] * output [:,5:6]

        expanded_output = np.hstack((bboxes, scores, cls))
        return expanded_output

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info['ratio']
        img = img_info['raw_img']
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]

        # resize
        bboxes /= ratio # divide by small ratio -> expand bboxes to raw_img's size
        cls = output[:,6]
        scores = output[:,4] * output[:,5]
        vis_res = vis(img,bboxes,scores,cls,cls_conf, self.cls_names)
        return vis_res

def main(exp, args):
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
    cap0 = cv2.VideoCapture(args.vid0_file if args.cam0_index is None else args.cam0_index)
    cap1 = cv2.VideoCapture(args.vid1_file if args.cam1_index is None else args.cam1_index)

    width0 = cap0.get(cv2.CAP_PROP_FRAME_WIDTH)
    width1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    if width0 != width1:
        logger.warning("Widths do not match. Using larger of the two.")
    width = max(width0, width1)

    height0 = cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if height0 != height1:
        logger.warning("Heights do not match.")
    height = height0 + height1

    fps0 = cap0.get(cv2.CAP_PROP_FPS)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    if fps0 != fps1:
        logger.warning("FPS does not match. Using lower of the two FPSs")
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
    sort_tracker = Sort()
    watchout_f = Watchout()
    watchout_r = Watchout()
    perspective = Perspective()

    while True:
        ret_val0, frame0 = cap0.read()
        ret_val1, frame1 = cap1.read()
        if not (ret_val0 and ret_val1):
            break
        else:
            frame0 = center_crop(frame0, args.crop0_width, args.crop0_height, nudge_down=-150)
            frame1 = center_crop(frame1, args.crop1_width, args.crop1_height, nudge_down=0)
            avgtimer.start('frame')

            outputs, img_info = predictor.inference(frame0, frame1)

            expanded_outputs = predictor.expand_bboxes(outputs, img_info)

            detections = expanded_outputs

            front_watchout_output = None
            rear_watchout_output = None
            if detections is not None:
                avgtimer.start('sort')
                tracked_output = sort_tracker.update(detections)
                avgtimer.end('sort')

                # split detections into front and rear, then run watchout separately
                front_dets, rear_dets = split_dets(tracked_output, args.crop0_width, args.crop0_height)
               #
                avgtimer.start('watchout')
                front_watchout_output = watchout_f.step(front_dets)
                rear_watchout_output = watchout_r.step(rear_dets)
                avgtimer.end('watchout')

                # then combine the split detections back into one array for visualization.
                combined_dets = combine_dets(front_watchout_output, rear_watchout_output, args.crop0_height)

            img = img_info['raw_img']
            if outputs[0] is None:
                result_frame = img
            else:
                output = outputs[0].cpu()
                bboxes = combined_dets[:,0:4]
                cls = combined_dets[:,4]
                scores = output[:,4] * output[:,5]
                cls_conf = 0.35
                distance = combined_dets[:,9]
                track_id = combined_dets[:,8]

                #result_frame = vis(img,bboxes,scores,cls,cls_conf, predictor.cls_names)
                result_frame = custom_vis(img,bboxes,scores,cls, distance, track_id, cls_conf, predictor.cls_names)

                perspective_output = perspective.step(combined_dets)

                for persp in perspective_output:
                    print(f"x: {persp[10]} , y: {persp[11]}")

            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch==27 or ch==ord('q') or ch==ord('Q'):
                break

            avgtimer.end('frame')
            logger.info('\n')
            logger.info(f"{1/(avgtimer.rolling_avg('frame'))} FPS")
            logger.info(f"SORT: {avgtimer.rolling_avg('sort')} seconds")
            logger.info(f"Watchout: {avgtimer.rolling_avg('watchout')} seconds")
            logger.info(f"{len(outputs[0]) if outputs[0] is not None else 0} objects detected")

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
