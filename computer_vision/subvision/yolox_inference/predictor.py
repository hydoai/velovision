import sys
sys.path.append('../..')
from PARAMETERS import HYDO_CLASSES


from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess, vis

import numpy as np
import torch

class Predictor:
    def __init__(
            self,
            model,
            exp,
            cls_names = HYDO_CLASSES,
            trt_file = None,
            decoder = None,
            device = 'gpu',
            fp16 = True,
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
            x = x.half() # DEBUG: i tried adding this to fix
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
