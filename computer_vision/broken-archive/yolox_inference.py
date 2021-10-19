import os
import time
from loguru import logger

import cv2
import torch
import numpy as np

#from yolox.data.data_augment import ValTransform
# has been moved for development to
from data_augment import ValTransform


from yolox.utils import postprocess, vis

from avgtimer import AvgTimer

# Classes
HYDO_CLASSES = ['bicycle', 'bus', 'car', 'cyclist', 'motorcycle', 'pedestrian', 'truck']

class Predictor():
    def __init__(self,
            model,
            exp,
            cls_names=HYDO_CLASSES,
            trt_file = None,
            decoder = None,
            device = 'gpu',
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
        self.preproc = ValTransform(legacy=False)

        self.avgtimer = AvgTimer()

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1,3, exp.test_size[0], exp.test_size[1]).cuda()

            self.model(x)
            self.model = model_trt
    def inference(self, img, img2 = None):
        # assuming img2 has all the same characteristics as img
        self.avgtimer.start('model-preproc')
        img_info = {"id":0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        if img2 is None:
            img_info["raw_img"] = img
        else:
            img_info["raw_img"] = np.append(img, img2, axis=0)

        if img2 is None:
            ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        else:
            ratio = self.test_size[0] / img.shape[0] / 2 # assuming horizontal images and same shape of the two imgs
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, img2, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        self.avgtimer.start("pin-mem")
        #img = img.pin_memory() # sometimes faster cpu-gpu transfer
        self.avgtimer.end("pin-mem")
        logger.info(f"Pin memory: {self.avgtimer.rolling_avg('pin-mem')}")

        img = img.float()

        if self.device == "gpu":

            self.avgtimer.start('cuda-copy')
           # img = img.cuda()
            # FASTER alternative to above: pinned non_blocking transfer
            #torch.cuda.synchronize()
            img = img.to(torch.device('cuda'), non_blocking=True) # 3ms
            self.avgtimer.end('cuda-copy')
            logger.info(f"CUDA copy: {self.avgtimer.rolling_avg('cuda-copy')}")

            if self.fp16:
                img = img.half()

        self.avgtimer.end('model-preproc')
        logger.info(f"Model pre-processing: {self.avgtimer.rolling_avg('model-preproc')}")
        with torch.no_grad():
            self.avgtimer.start('model-only')
            outputs = self.model(img)
            self.avgtimer.end('model-only')

            logger.info(f"Model only: {self.avgtimer.rolling_avg('model-only')}")
            
            self.avgtimer.start('model-postproc')
            if self.decoder is not None: 
                self.avgtimer.start('decoder')
                outputs = self.decoder(outputs, dtype=outputs.type())
                self.avgtimer.end('decoder')
                logger.info(f"Decoder: {self.avgtimer.rolling_avg('decoder')}")
            self.avgtimer.start('postprocess')
            outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)
            self.avgtimer.end('postprocess')
            logger.info(f"postprocess: {self.avgtimer.rolling_avg('postprocess')}")
            self.avgtimer.end('model-postproc')
            logger.info(f"Model postprocessing: {self.avgtimer.rolling_avg('model-postproc')}")
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        cls = output[:,6]
        scores = output[:,4] * output[:,5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

