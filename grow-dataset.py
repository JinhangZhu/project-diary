import glob
import os
import shutil

import cv2
# import matplotlib.pyplot as plt
import numpy as np
# import torch
from tqdm import tqdm

from differentiate import *
from yolov3.utils.utils import xywh2xyxy  # , xyxy2xywh

CLS = ['hand']
NEW_CLS = ['left_hand', 'right_hand']


DS_PATH = '../datasets/ego-hand'
img_paths = sorted(glob.glob(DS_PATH + '/images/' + '*.jpg'))
ann_paths = sorted(glob.glob(DS_PATH + '/labels/' + '*.txt'))

# New folder
diff_labels_path = os.path.join(DS_PATH, 'diff_labels_2nd')
if os.path.exists(diff_labels_path):
    pass
    # shutil.rmtree(diff_labels_path)  # delete output folder
else:
    os.makedirs(diff_labels_path)  # make new output folder

# RESTART
diff_paths = sorted(glob.glob(diff_labels_path + '/' + '*txt'))
restart_index = len(diff_paths)
img_paths = img_paths[restart_index:]
ann_paths = ann_paths[restart_index:]

# Labels
with open(os.path.join(DS_PATH, 'diff.names'), 'w') as f:
    for n in NEW_CLS:
        f.write('%s\n' % n)

for ip, ap in zip(img_paths, ann_paths):
    # Read original annotation
    img = cv2.imread(ip)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = get_ann_boxes(ap)
    np_boxes = np.asarray(boxes)
    np_boxes[:, 1:] = xywh2xyxy(np_boxes[:, 1:])
    np_boxes[:, 1:] = np.clip(np_boxes[:, 1:], 0, 1)    # IMPORTANT
    np_boxes[:, 1:] = scale_xyxy(np_boxes[:, 1:], img.shape[:2])
    irt_np_boxes = np.insert(np_boxes, 1, 1, axis=1)

    # Info output
    print(('%20s: %s' * 2) % ('Image', os.path.basename(ap)[:-4] + '.jpg', 'Hands', np_boxes.shape[0]))

    # Differentiation
    lr_boxes = diff_hands(img, irt_np_boxes, extend_scale=0, color_thres=10, bin_thres=5, max_region=False, view=False)
    np_boxes = np.asarray(boxes)
    np_boxes[:, 0] = lr_boxes[:, 0]

    # Write new annotation
    new_ann_path = os.path.join(diff_labels_path, os.path.basename(ap))
    with open(new_ann_path, 'w') as f:
        for np_box in np_boxes:
            f.write(('%g ' * 5 + '\n') % (*np_box[:],))
