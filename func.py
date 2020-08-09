# Useful functions for the project

import glob
# Libraries
import os
import shutil

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tabulate import tabulate


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
        plt.scatter(x[:, 0], x[:, 1], label='True position')
        plt.savefig(save_path + '/True position.png')

    # Kmeans clustering
    kmeans = KMeans(n_clusters=k_clusters).fit(x)
    anchors = kmeans.cluster_centers_
    anchors = anchors * img_size

    # Plot scatter graph of [w, h] pairs
    if save_path:
        plt.figure()
        plt.scatter(x[:, 0]*img_size, x[:, 1]*img_size, c=kmeans.labels_, cmap='viridis')
        plt.scatter(anchors[:, 0], anchors[:, 1], color='#a23500')
        plt.title("Width-height Pair Position (Scaled to {}*{})".format(img_size, img_size))
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.savefig(save_path + '/anchor-kmeans.png')

    anchors = np.rint(anchors)

    # Plot anchors
    if save_path:
        fig, ax = plt.subplots()
        for k in range(k_clusters):
            rect = patches.Rectangle(
                (img_size/2 - anchors[k, 0]/2, img_size/2 - anchors[k, 1]/2),
                anchors[k, 0], anchors[k, 1],
                linewidth=1,
                edgecolor='b',
                facecolor='b',
                fill=None
            )
            ax.add_patch(rect)
        ax.set_aspect(1.0)
        plt.axis([0, img_size, 0, img_size])
        plt.title("Anchor Boxes (Scaled to {}*{})".format(img_size, img_size))
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.savefig(save_path + "/anchor-boxes-rects.png")

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
