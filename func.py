# Useful functions for the project

import glob
# Libraries
import os
import random
import shutil

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tabulate import tabulate

from yolov3.utils.utils import plot_one_box, xywh2xyxy

def verify_dataset(dataset_path, dpi=150):
    """Randomly display an image with its plotted bounding boxes.
    """
    img_paths = sorted(glob.glob(dataset_path + '/images/' + '*.jpg'))
    ann_paths = sorted(glob.glob(dataset_path + '/labels/' + '*.txt'))
    cls_path = glob.glob(dataset_path + '/*.names')

    classes = []
    with open(cls_path[0], 'r') as f:
        for line in f:
            classes.append(line[:-1])
    
    i = random.randint(0, len(img_paths)-1)
    demo = verify_annotation(img_paths[i], ann_paths[i], classes)
    print(
        '\nImage: ', img_paths[i],
        '\nLabel: ', ann_paths[i]
    )
    demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
    plt.figure(dpi=dpi)
    plt.imshow(demo)
    plt.show()

def verify_annotation(img_path, ann_path, labels):
    img = cv2.imread(img_path)
    boxes = []  # xywh
    with open(ann_path, 'r') as f:
        for line in f:
            line = line[:-1].split()
            line[0] = int(line[0])
            line[1:] = [float(i) for i in line[1:]]
            boxes.append(line)
    boxes = np.asarray(boxes)
    boxes[:, 1:] = xywh2xyxy(boxes[:, 1:])  # xyxy
    if boxes[0, 1] <= 1:
        boxes[:, 1::2] *= img.shape[1]
        boxes[:, 2::2] *= img.shape[0]

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(labels))]
    for box in boxes:
        label = '%s' % (labels[int(box[0])])
        plot_one_box(box[1:], img, label=label, color=colors[int(box[0])])
    return img


def kmeans_anchors(train_path='../datasets/ego-hand/train.txt', k_clusters=9, img_size=416, save_path=None):
    """Generate anchors for the dataset.
    Normalised labels: cls id, center x, center y, width, height
    """

    # Get paths of training images and labels
    ann_paths = []
    train_name = os.path.basename(train_path)
    ds_path = train_path[:-len(train_name)]
    with open(train_path, 'r') as f:
        for line in f:
            line = line[:-1]
            img_name = os.path.basename(line)
            ann_path = os.path.join(ds_path + 'labels', img_name[:-3] + 'txt')
            ann_paths.append(ann_path)

    # Get NORMALISED widths and heights from annotation files *.txt
    ws = []
    hs = []

    for ann_path in ann_paths:
        with open(ann_path, 'r') as f:
            for line in f:
                line = line[:-1].split()
                w, h = [float(i) for i in line[-2:]]
                ws.append(w)
                hs.append(h)

    # Generate input data as [w, h] pairs
    ws = np.asarray(ws)
    hs = np.asarray(hs)
    x = [ws, hs]
    x = np.asarray(x).transpose()

    # Plot the [w, h] pairs in scatter graph
    if save_path:
        # New folder
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        plt.figure(dpi=300)
        plt.scatter(x[:, 0], x[:, 1], label='True position')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.savefig(save_path + '/True position.pdf')

    # Kmeans clustering
    kmeans = KMeans(n_clusters=k_clusters).fit(x)
    anchors = kmeans.cluster_centers_
    anchors = anchors * img_size

    # Plot scatter graph of [w, h] pairs
    if save_path:
        plt.figure(dpi=300)
        plt.scatter(x[:, 0], x[:, 1], c=kmeans.labels_, cmap='viridis')
        plt.scatter(anchors[:, 0]/img_size, anchors[:, 1]/img_size, color='#a23500')
        # plt.title("Width-height Pair Position")
        plt.xlabel('Width')
        plt.ylabel('Height')
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        plt.savefig(save_path + '/anchor-kmeans-ori.pdf')

        plt.figure(dpi=300)
        plt.scatter(x[:, 0]*img_size, x[:, 1]*img_size, c=kmeans.labels_, cmap='viridis')
        plt.scatter(anchors[:, 0], anchors[:, 1], color='#a23500')
        # plt.title("Width-height Pair Position (Scaled to {}*{})".format(img_size, img_size))
        plt.xlabel('Width')
        plt.ylabel('Height')
        # plt.xlim((0, img_size))
        # plt.ylim((0, img_size))
        plt.savefig(save_path + '/anchor-kmeans.pdf')

    anchors = np.rint(anchors)

    # Plot anchors
    if save_path:
        fig, ax = plt.subplots(dpi=300)
        for k in range(k_clusters):
            rect = patches.Rectangle(
                (img_size/2 - anchors[k, 0]/2, img_size/2 - anchors[k, 1]/2),
                anchors[k, 0], anchors[k, 1],
                linewidth=1,
                edgecolor='tab:blue',
                facecolor='tab:blue',
                fill=None
            )
            ax.add_patch(rect)
        ax.set_aspect(1.0)
        plt.axis([0, img_size, 0, img_size])
        # plt.title("Anchor Boxes (Scaled to {}*{})".format(img_size, img_size))
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.savefig(save_path + "/anchor-boxes-rects.pdf")

    # Print and save anchors
    anchors.sort(axis=0)
    anchors = anchors.astype(int)
    print("Anchors are: \n{}".format(anchors))

    if save_path:
        with open(os.path.join(ds_path, 'anchors.txt'), 'w') as f:
            for w, h in anchors:
                f.write("{}, {}\n".format(w, h))

        print("\nAnchors saved to {}".format(os.path.join(ds_path, 'anchors.txt')))

    return anchors

def compare_anchors(hand_anchors, coco_anchors, img_size=416, save_path=None):
    if isinstance(hand_anchors, list):
        hand_anchors = np.asarray(hand_anchors, dtype=int)
    if isinstance(coco_anchors, list):
        coco_anchors = np.asarray(coco_anchors, dtype=int)

    fig, ax = plt.subplots(dpi=300)
    for k in range(9):
        rect = patches.Rectangle(
            (img_size/2 - coco_anchors[k, 0]/2, img_size/2 - coco_anchors[k, 1]/2),
            coco_anchors[k, 0], coco_anchors[k, 1],
            linewidth=1,
            edgecolor='tab:orange',
            facecolor='tab:orange',
            fill=None,
            label = 'COCO'
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            (img_size/2 - hand_anchors[k, 0]/2, img_size/2 - hand_anchors[k, 1]/2),
            hand_anchors[k, 0], hand_anchors[k, 1],
            linewidth=1,
            edgecolor='tab:blue',
            facecolor='tab:blue',
            fill=None,
            label = 'Epichands'
        )
        ax.add_patch(rect)
        if k == 0:
            ax.legend(loc='best')
    ax.set_aspect(1.0)
    plt.axis([0, img_size, 0, img_size])
    # plt.title("Anchor Boxes (Scaled to {}*{})".format(img_size, img_size))
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()
    if save_path:
        plt.savefig(save_path + "/compare-anchor-boxes-rects.pdf")
    