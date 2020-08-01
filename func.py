# Useful functions for the project

# Libraries
import os
import shutil

import cv2
import numpy as np
from tabulate import tabulate


def crop_bbox(img, boxes):
    if isinstance(boxes, list):
        boxes = np.asarray(boxes, dtype=int)

    xyxy = boxes[:, 1:]
    crops = []
    for i in range(boxes.shape[0]):
        crop_img = img[xyxy[i, 1]:xyxy[i, 3], xyxy[i, 0]:xyxy[i, 2]]
        crops.append(crop_img)

    return crops


def boxes_info(boxes, labels):
    for i, box in enumerate(boxes):
        label = labels[box[0]]
        xywh = box[1:]
        print(
            '\nNormalised bounding box {}: {}\n'.format(i, label),
            tabulate(
                [[xywh[0], xywh[1], xywh[2], xywh[3]]],
                headers=['LT_x', 'LT_y', 'RB_x', 'RB_y']
            )
        )
