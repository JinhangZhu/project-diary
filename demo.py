import cv2
import numpy as np
from func import verify_labels
import matplotlib.pyplot as plt
import glob
import random

# from differentiate import diff_hands, boxes_info

# CLS = ['hand']
NEW_CLS = ['left_hand', 'right_hand']
# IMAGE = 'yolov3/data/samples/0000004831.jpg'

# # IMPORTANT
# boxes = np.array(
#     [
#         [0, 0.8236, 1119, 741, 1315, 960],
#         [0, 0.2, 350, 773, 500, 900],
#         [0, 0.74608, 620, 734, 846, 1034]
#     ]
# )

# # IMPORTANT
# img = cv2.imread(IMAGE)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ret = diff_hands(img, boxes)
# print(boxes_info(ret, NEW_CLS))

PATH = '../datasets/epichands'

img_paths = sorted(glob.glob(PATH + '/images/' + '*.jpg'))
ann_paths = sorted(glob.glob(PATH + '/labels/' + '*.txt'))

i = random.randint(0, len(img_paths)-1)
demo = verify_labels(img_paths[i], ann_paths[i], NEW_CLS)
print(
    '\nImage: ', img_paths[i],
    '\nLabel: ', ann_paths[i]
)
demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
plt.imshow(demo)
plt.show()
