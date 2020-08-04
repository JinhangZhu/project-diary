import cv2
import numpy as np

from differentiate import diff_hands, boxes_info

CLS = ['hand']
NEW_CLS = ['left_hand', 'right_hand']
IMAGE = 'yolov3/data/samples/0000004831.jpg'

# IMPORTANT
boxes = np.array(
    [
        [0, 0.8236, 1119, 741, 1315, 960],
        [0, 0.2, 350, 773, 500, 900],
        [0, 0.74608, 620, 734, 846, 1034]
    ]
)

# IMPORTANT
img = cv2.imread(IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

ret = diff_hands(img, boxes)
print(boxes_info(ret, NEW_CLS))
