import pandas as pd
import numpy as np
import cv2
from kitti.utils import get_pointcloud, get_transform_matrix
from tools.utils import CONFIG
import os
from TrafficDetector import TrafficDetector
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy import stats

DATASET_PATH = 'kitti'
vld_path = os.path.join(DATASET_PATH, "velodyne/{:06d}.bin")
img_path = os.path.join(DATASET_PATH, "image_2/{:06d}.png")
clb_path = os.path.join(DATASET_PATH, "calib/{:06d}.txt")

frame_num = 6

points, raw_points = get_pointcloud(vld_path.format(frame_num))

# Get transforms and image
P2, R0, Tr = get_transform_matrix(clb_path.format(frame_num))
image = cv2.imread(img_path.format(frame_num))

raw_points = raw_points[:, raw_points[-1, :] > 0]

# Project 3d points
pts3d_cam = R0 @ Tr @ raw_points
mask = pts3d_cam[2, :] >= 0  # Z >= 0
pts3d = np.transpose(raw_points[:, mask])
pts2d_cam = P2 @ pts3d_cam[:, mask]
pts2d = (pts2d_cam / pts2d_cam[2, :])[:-1, :].T

# Draw points on image
for point in pts2d:
    cv2.circle(img=image,
               center=(int(point[0]), int(point[1])),
               radius=0,
               color=(255, 255, 255),
               thickness=-1)

# detect objects
detector = TrafficDetector()
objects, image = detector.traffic_detections(image)

# Highlight object points
mask = ((pts2d[:, 1] > objects[0][0][0][0]) &
        (pts2d[:, 1] < objects[0][0][0][2]) &
        (pts2d[:, 0] > objects[0][0][0][1]) &
        (pts2d[:, 0] < objects[0][0][0][3]))

for point in pts2d[mask]:
    cv2.circle(img=image,
               center=(int(point[0]), int(point[1])),
               radius=0,
               color=(0, 0, 255),
               thickness=1)

# Cluster
pts3d_fru = pts3d[mask]

db = DBSCAN(eps=0.4).fit(pts3d_fru[:, :2])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

largest_cluster = stats.mode(labels)

# Draw birds eye
birds_eye_scale = 5
birds_eye = np.zeros(shape=[
    CONFIG['max distance'] * 2 * birds_eye_scale,
    CONFIG['max distance'] * 2 * birds_eye_scale, 3
],
                     dtype=np.uint8)
for index, row in points.iterrows():
    cv2.circle(img=birds_eye,
               center=(int(row['x'] * birds_eye_scale +
                           CONFIG['max distance'] * birds_eye_scale),
                       int(row['y'] * -birds_eye_scale +
                           CONFIG['max distance'] * birds_eye_scale)),
               radius=0,
               color=(255, 255, 255),
               thickness=-1)

# Highlight frustum & object cluster
for i in range(len(labels)):
    if labels[i] == largest_cluster[0]:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.circle(img=birds_eye,
               center=(int(pts3d_fru[i][0] * birds_eye_scale +
                           CONFIG['max distance'] * birds_eye_scale),
                       int(pts3d_fru[i][1] * -birds_eye_scale +
                           CONFIG['max distance'] * birds_eye_scale)),
               radius=0,
               color=color,
               thickness=-1)
    cv2.circle(img=image,
               center=(int(pts2d[mask][i][0]), int(pts2d[mask][i][1])),
               radius=0,
               color=color,
               thickness=-1)

cv2.imshow('Frustum', birds_eye)
cv2.imshow('Camera', image)

cv2.waitKey()