import numpy as np
import cv2

from yolox.utils.visualize import _COLORS

def TTE_EQ_custom_vis(img, boxes, scores, cls_ids, distance, track_id, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        distance_value = distance[i]
        track_value = track_id[i]
        text = f"{class_names[cls_id]} {int(track_value)} : {round(distance_value,1)} m"
        txt_color = (0,0,0) if np.mean(_COLORS[cls_id]) > 0.5 else (255,255,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4 , 1)[0]
        cv2.rectangle(img, (x0,y0), (x1,y1), color ,2)
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
                img,
                (x0,y0+1),
                (x0 + txt_size[0] + 1 , y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
                )
        cv2.putText(img, text, (x0,y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
