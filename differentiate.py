# Libraries
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from yolov3.utils.utils import plot_one_box


def diff_hands(img, boxes):
    """Differentiate the predited bounding boxes into two labels. <left_hand>, <right_hand>
    Arguments:
        img: image
        boxes: predicted bounding boxes, format: (ID, centre_x, centre_y, width, height)
        #ellipse_info: list of ((x, y), angle) of the fitted ellipse
    Return:
        diff_boxes: differentiated bounding boxes, format: (ID, centre_x, centre_y, width, height), where ID relates to new 'labels'.
    """
    # Info
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)
    n_hands = boxes.shape[0]

    if n_hands == 0:
        print("No hands detected.")
        return 0

    elif n_hands == 1:
        # Check extended boxes
        extracted, exd_boxes = extractHandArm(img, boxes, open_size=5, extend_scale=0.5, color_thres=20)
        ellipse_info = ellipseFit(extracted, bin_thres=10)
        angle = ellipse_info[0][1]
        boxes[0][0] = identifyOneHand(angle)

    elif n_hands == 2:
        # Check a-little-bit extended boxes
        extracted, exd_boxes = extractHandArm(img, boxes, open_size=5, extend_scale=0.2, color_thres=20)
        ellipse_info = ellipseFit(extracted, bin_thres=10)
        _, angle1 = ellipse_info[0]
        _, angle2 = ellipse_info[1]

        # Flag that angles are 0-90 and 90-180 separately.
        isAngleRight1 = identifyOneHand(angle1)
        isAngleRight2 = identifyOneHand(angle2)
        isAngleDiff = isAngleRight1 ^ isAngleRight2

        if not isAngleDiff:
            c_x0 = (boxes[0][1] + boxes[0][3]) / 2  # center x0
            c_x1 = (boxes[1][1] + boxes[1][3]) / 2  # center x1
            boxes[0][0] = int(c_x0 > c_x1)
            boxes[1][0] = int(c_x0 < c_x1)
        else:
            boxes[0][0] = isAngleRight1
            boxes[1][0] = isAngleRight2

    else:   # More hands detected
        # Sort the bounding boxes according to the confidences
        boxes = boxes[boxes[:, 1].argsort()[::-1]]  # Descending order
        boxes = boxes[:2, :]
        boxes = diff_hands(img, boxes)   # Recursive step for once

    return boxes


def identifyOneHand(angle):
    """Identify one hand as left/right hand.
    Arguments:
        #shape: image.shape
        #pos: (x, y)
        angle: degree
    Returns:
        0 for left_hand, 1 for right_hand
    """
    # x, y = pos      # positions
    # w = shape[1]    # Width
    # ct_axis = w/2   # center axis

    if angle <= 90:
        return 0
    else:
        return 1


def extractHandArm(img, boxes, open_size=3, extend_scale=1, color_thres=40, view=False):
    """Extract hands and arms around them from bounding boxes and the image.
    Arguments:
        img: numpy array
        boxes: list of numpy arrays, format: (cls, conf, ltx, lty, rbx, rby)
    Returns:
        res: list of numpy arrays, each array stands for an processed crop with skin ares kept.
        exd_boxes: numpy array, coordinates of returns, (ltx, lty, rbx, rby)
    """
    # 获取裁片
    crops = crop_bbox(img, boxes)
    # 裁片扩展框
    exd_crops, exd_boxes = extendRegion(img, boxes, extend_scale)

    # 返回提取框
    res = []

    for crop, exd_crop in zip(crops, exd_crops):
        # 去噪
        # crop = cv2.blur(crop, (3, 3))  # 均值滤波
        # crop = cv2.GaussianBlur(crop_blur, (3, 3), 0)  # 高斯滤波
        # crop = cv2.medianBlur(crop_blur, 5)    # 中值滤波

        # 颜色模型阈值分割
        skin_masked_crop = skinMask(crop)
        # plt.imshow(skin_masked_crop)
        # plt.show()

        # 形态学开运算
        opened_skin_crop = openOperation(skin_masked_crop, open_size)
        # plt.imshow(opened_skin_crop)
        # plt.show()

        # 提取平均颜色
        extract_mask = opened_skin_crop != [0, 0, 0]
        extract_region = opened_skin_crop[extract_mask[:, :, 0], :]
        skin_avg = np.mean(extract_region, axis=0)

        # 颜色阈值筛
        color_masked_crop = colorMask(exd_crop, skin_avg, color_thres)
        # plt.imshow(color_masked_crop)
        # plt.show()

        # 颜色模型阈值分割
        skin_masked_exd_crop = skinMask(color_masked_crop)
        # plt.imshow(skin_masked_exd_crop)
        # plt.show()

        # 形态学开运算
        opened_skin_masked_exd_crop = openOperation(skin_masked_exd_crop)
        # plt.imshow(opened_skin_masked_exd_crop)
        # plt.show()

        if view:
            plt.figure(figsize=(20, 16))
            plt.subplot(171)
            plt.imshow(crop)
            plt.title('Original Crop')
            plt.subplot(172)
            plt.imshow(skin_masked_crop)
            plt.title('1st Thresholding')
            plt.subplot(173)
            plt.imshow(opened_skin_crop)
            plt.title('1st Open Operation')
            plt.subplot(174)
            plt.imshow(exd_crop)
            plt.title('Extended Crop')
            plt.subplot(175)
            plt.imshow(color_masked_crop)
            plt.title('Color approximation')
            plt.subplot(176)
            plt.imshow(skin_masked_exd_crop)
            plt.title('2nd Thresholding')
            plt.subplot(177)
            plt.imshow(opened_skin_masked_exd_crop)
            plt.title('2nd Open Operation')
            plt.show()

        res.append(opened_skin_masked_exd_crop)

    return res, exd_boxes


# 方法一：采用KMeans clustering获取手部
def kmeansMask(region, k=3):
    """KMeans Clustering on the region to get the hand.

    Reference: https://morioh.com/p/b6763f7527d5

    Returns:
        masked_region:
        avg_color:
    """
    # 将3D的图片reshape为2D的array，即宽高二维上压平为一维
    pixel_values = region.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # print('After flattening: ', pixel_values.shape)

    # 停止策略
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # kmeans clustering
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 坐标类型恢复
    centers = np.uint8(centers)

    segmented_region = centers[labels.flatten()]  # 配对对应label
    segmented_region = segmented_region.reshape(region.shape)  # 恢复size
    # plt.imshow(segmented_region)
    # plt.show()

    # 找到最大的cluster，也就是出现次数最多的label
    most_label = np.bincount(labels.flatten()).argmax()
    # print('Most frequently: %s' % most_label)

    # 选择性显示/掩膜显示
    colors = [[random.randint(200, 255) for _ in range(3)] for _ in range(3)]

    masked_region = np.copy(region)
    masked_region = masked_region.reshape((-1, 3))
    mask = labels == most_label
    # masked_region[mask[:, 0], :] = colors[1]  # 最大的区域染色
    masked_region[~mask[:, 0], :] = [0, 0, 0]  # 其余统一为黑色
    masked_region = masked_region.reshape(region.shape)
    # plt.imshow(masked_region)
    # plt.show()

    # 找到手的平均颜色（即cluster的平均颜色）
    most_mask = labels == most_label
    most_avg = np.mean(pixel_values[most_mask[:, 0], :], axis=0, dtype=np.int32)
    # color_demo = np.tile(most_avg, 10000).reshape((100, 100, 3))

    return masked_region, most_avg

# 方法二： YCrCb颜色空间Cr分量+OTSU法阈值分割算法
# References: https://blog.csdn.net/qq_41562704/article/details/88975569


def skinMask(region):
    YCrCb = cv2.cvtColor(region, cv2.COLOR_RGB2YCrCb)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
    res = cv2.bitwise_and(region, region, mask=skin)
    return res


def openOperation(region, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 设置卷积核
    erosion = cv2.erode(region, kernel)  # 腐蚀操作
    dilation = cv2.dilate(erosion, kernel)  # 膨胀操作
    return dilation

# 手部扩展：将手部扩展到能够包含手臂


def extendRegion(img, boxes, extend_scale=1):
    if isinstance(boxes, list):
        boxes = np.asarray(boxes, dtype=np.float32)
    xyxy = boxes[:, 2:].astype(int)
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    exd_x0 = np.clip(xyxy[:, 0] - (w*extend_scale).astype(int), 0, img.shape[1])
    exd_y0 = np.clip(xyxy[:, 1] - (h*extend_scale).astype(int), 0, img.shape[0])
    exd_x1 = np.clip(xyxy[:, 2] + (w*extend_scale).astype(int), 0, img.shape[1])
    exd_y1 = np.clip(xyxy[:, 3] + (h*extend_scale).astype(int), 0, img.shape[0])

    exd_crops = []
    for i in range(boxes.shape[0]):
        crop_img = img[exd_y0[i]:exd_y1[i], exd_x0[i]:exd_x1[i]]
        exd_crops.append(crop_img)

    exd_boxes = np.concatenate(
        (boxes[:, :2], exd_x0.reshape((2, 1)), exd_y0.reshape((2, 1)), exd_x1.reshape((2, 1)), exd_y1.reshape((2, 1))),
        axis=1
    )

    return exd_crops, exd_boxes

# 颜色限制


def colorMask(region, color, thres=50):
    flatten_region = (region - color).reshape((-1, 3))
    near_mask = np.linalg.norm(flatten_region, axis=1) < 50

    masked_region = np.copy(region)
    masked_region = masked_region.reshape((-1, 3))
    masked_region[~near_mask, :] = [0, 0, 0]
    masked_region = masked_region.reshape(region.shape)

    return masked_region


def crop_bbox(img, boxes):
    if isinstance(boxes, list):
        boxes = np.asarray(boxes)

    xyxy = boxes[:, 2:].astype(int)
    crops = []
    for i in range(boxes.shape[0]):
        crop_img = img[xyxy[i, 1]:xyxy[i, 3], xyxy[i, 0]:xyxy[i, 2]]
        crops.append(crop_img)

    return crops


def boxes_info(boxes, labels):
    for i, box in enumerate(boxes):
        label = labels[int(box[0])]
        conf = box[1]
        xywh = box[2:]
        print(
            '\nNormalised bounding box {}: {}\n'.format(i, label),
            tabulate(
                [[conf, xywh[0], xywh[1], xywh[2], xywh[3]]],
                headers=['conf', 'LT_x', 'LT_y', 'RB_x', 'RB_y']
            )
        )


def get_boxes(path):
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1].split()
            line[0] = int(line[0])
            line[1:] = [float(i) for i in line[1:]]
            boxes.append(line)
    return boxes


def ellipseFit(regions, bin_thres=10, view=False):
    """Fit ellipses to the objects in the regions.
    Arguments:
        regions: numpy array in RGB, contains objects
        bin_thres: threshold for binary thresholding
        view: whether to show the images
    Returns:
        [((x, y), angle)]: float, center x, center y, orientation agle
    """
    res = []

    # Colors
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(regions))]

    for i, region in enumerate(regions):
        # Copy
        rc = region.copy()

        # Plotting config
        tl = round(0.002 * (rc.shape[0] + rc.shape[1]) / 2) + 1  # Line thickness
        tf = max(tl-1, 1)   # Font thickness
        rd = tl

        # Find contours
        gray = cv2.cvtColor(rc, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, bin_thres, 255, cv2.THRESH_BINARY)
        contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]

        # Ellipse fitting
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(rc, ellipse, colors[i], 2)

        # Info text
        x, y, angle = int(ellipse[0][0]), int(ellipse[0][1]), int(ellipse[2])
        angle = rectifyAngle(angle)
        res.append(((x, y), angle))
        label = '({}. {}), {} deg'.format(x, y, angle)

        # Annotation
        cv2.circle(rc, (x, y), rd, [225, 255, 255], -11)
        cv2.putText(rc, label, (x - 100, y - 20), 0, tl/3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        # Plotting
        if view:
            plt.imshow(rc)
            plt.show()
    return res


def rectifyAngle(angle):
    """Rectify angle from cv2 numpy array to intuitions.
    """
    return -angle+90+180*(angle > 90)


def save_result(boxes, img, labels, save_path):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(boxes))]
    # boxes = scale_coords(torch.Size([320, 512]), boxes, img.shape).round()
    for box in boxes:
        label = '%s %.2f' % (labels[int(box[0])], box[1])
        plot_one_box(box[2:], img, label=label, color=colors[int(box[0])])
    cv2.imwrite(save_path, img)


def isOverlap1D(box1, box2):
    """Check if two 1D boxes overlap.
    Reference: https://stackoverflow.com/a/20925869/12646778
    Arguments:
        box1, box2: format: (xmin, xmax)
    Returns:
        res: bool, True for overlapping, False for not
    """
    xmin1, xmax1 = box1
    xmin2, xmax2 = box2
    return xmax1 >= xmin2 and xmax2 >= xmin1


def isOverlap2D(box1, box2):
    """Check if the two 2D boxes overlap.
    Arguments:
        box1, box2: format: (ltx, lty, rbx, rby)
    Returns: 
        res: bool, True for overlapping, False for not
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    return isOverlap1D((xmin1, xmax1), (xmin2, xmax2)) and isOverlap1D((ymin1, ymax1), (ymin2, ymax2))
