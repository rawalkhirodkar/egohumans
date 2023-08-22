import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from json import JSONEncoder
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import linalg


###--------------------------------------------------------------
def read_points(file, target_person_idx, num_keypoints=17):
    points_2d = []
    if not os.path.exists(file):
        return np.zeros((num_keypoints, 2))

    with open(file, 'r') as f:
        data = f.readlines()
        data = [x.replace('\n', '') for x in data]

    annotated_person_ids = [int(x) for x in data[::num_keypoints + 1]]

    if target_person_idx not in annotated_person_ids:
        return np.zeros((num_keypoints, 2))

    idx_ = (num_keypoints + 1) * annotated_person_ids.index(target_person_idx)
    data = data[idx_:idx_ + num_keypoints + 1]

    ##-----drop the person idx----
    data = data[1:]

    for point in data:
        point = point.strip().split(',')
        point = np.array([int(x) for x in point]).reshape(1, -1)
        points_2d.append(point)

    points_2d = np.concatenate(points_2d, axis=0)

    return points_2d


###--------------------------------------------------------------
## reference: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
def triangulation(multiview_evidence):
    num_cameras = len(multiview_evidence)
    assert(num_cameras > 1)
    num_points = len(multiview_evidence[0]['points_2d'])

    ## we get the 3D rays from camera center in cam coordinates for the 2d points 
    for view_idx in range(num_cameras): 
        singleview_evidence = multiview_evidence[view_idx]

        cam = singleview_evidence['cam']
        points_2d = singleview_evidence['points_2d'] ## in image coordinates

        rays_3d = []

        for point_idx in range(num_points):
            point_2d = points_2d[point_idx]
            ray_3d = cam.cam_from_image(point_2d=point_2d)
            assert(len(ray_3d) == 3 and ray_3d[2] == 1)            
            rays_3d.append(ray_3d[:2].reshape(1, -1)) ## drop the last value, which is 1

        rays_3d = np.concatenate(rays_3d, axis=0) ## N x 2
        assert(rays_3d.shape[1] == 2)

        multiview_evidence[view_idx]['rays_3d'] = rays_3d

    triangulated_points_3d = direct_linear_transform(multiview_evidence)

    return triangulated_points_3d


###--------------------------------------------------------------
def direct_linear_transform(multiview_evidence):
    num_cameras = len(multiview_evidence)
    assert(num_cameras > 1)
    num_points = len(multiview_evidence[0]['points_2d'])

    triangulated_points_3d = []
    ## loop over each multi-view correspondence
    for point_idx in range(num_points):

        ##------create matrix A--------
        A = np.zeros((2*num_cameras, 4))

        for cam_idx in range(num_cameras):
            singleview_evidence = multiview_evidence[cam_idx]
            cam = singleview_evidence['cam']
            ray_3d = singleview_evidence['rays_3d'][point_idx]
            extrinsics = cam.extrinsics ## 4 x 4, last row is [0, 0, 0, 1]

            A[2*cam_idx + 0] = ray_3d[1]*extrinsics[2] - extrinsics[1]
            A[2*cam_idx + 1] = extrinsics[0] - ray_3d[0]*extrinsics[2] 

        ##--------solve SVD-------------
        B = A.transpose() @ A
        U, S, Vh = linalg.svd(B, full_matrices=False)

        triangulated_point_3d = Vh[3,0:3]/Vh[3,3]
        triangulated_points_3d.append(triangulated_point_3d.reshape(1, -1))

    triangulated_points_3d = np.concatenate(triangulated_points_3d, axis=0)
    return triangulated_points_3d

###--------------------------------------------------------------












