import numpy as np
from typing import Union
import torch
import cv2

## returns T x points_3d
def linear_transform(points_3d, T):
    assert(points_3d.shape[1] == 3)

    points_3d_homo = np.ones((4, points_3d.shape[0]))
    points_3d_homo[:3, :] = np.copy(points_3d.T)

    points_3d_prime_homo = np.dot(T, points_3d_homo)
    points_3d_prime = points_3d_prime_homo[:3, :]/ points_3d_prime_homo[3, :]
    points_3d_prime = points_3d_prime.T
    return points_3d_prime

def distance_from_plane(point, plane):
    a, b, c, d = plane.get_equation()
    distance = np.abs(a*point[0] + b*point[1] + c*point[2] + d)/np.sqrt(a*a + b*b + c*c)
    return distance

def projected_point_to_plane(point, plane):
    a, b, c, d = plane.get_equation() ## ax+by+cz +d = 0
    distance = np.abs(a*point[0] + b*point[1] + c*point[2] + d)/np.sqrt(a*a + b*b + c*c)

    projected_point = point - distance*np.array([a, b, c])

    # assert(is_point_on_plane(projected_point, plane)) ## remove this, causing too many crashes

    return projected_point, distance

def is_point_on_plane(point, plane, thres=1e-3):
    a, b, c, d = plane.get_equation()
    val = a*point[0] + b*point[1] + c*point[2] + d

    if np.abs(val) <= thres:
        return True

    return False

def plane_unit_normal(plane):
    a, b, c, d = plane.get_equation()

    unit_normal = np.array([a, b, c])/np.sqrt(a*a + b*b + c*c)

    return unit_normal

def get_point_on_plane(plane):
    a, b, c, d = plane.get_equation()

    if a != 0:
        point = np.array([-d/a, 0, 0])
    elif b != 0:
        point = np.array([0, -d/b, 0])
    elif c != 0:
        point = np.array([0, 0, -d/c])
    else:
        raise ValueError("Plane is not valid")

    return point


# ##------------for loop---------------------
def fast_circle(image, overlay, points_2d, radius, color):
    for idx in range(len(points_2d)):
        image = cv2.circle(image, (round(points_2d[idx, 0]), round(points_2d[idx, 1])), radius, color, -1)

        if overlay is not None:
            overlay = cv2.circle(overlay, (round(points_2d[idx, 0]), round(points_2d[idx, 1])), radius, color, -1)

    if overlay is None:
        return image

    return image, overlay

## --verctorized version---------------------
def slow_circle(image, overlay, points_2d, radius, color):
    points_2d = points_2d.astype(np.int) ## convert to integer
    image_raw = np.zeros_like(image)
    image_raw[points_2d[:, 1], points_2d[:, 0], 0] = color[0]
    image_raw[points_2d[:, 1], points_2d[:, 0], 1] = color[1]
    image_raw[points_2d[:, 1], points_2d[:, 0], 2] = color[2]

    image_raw[:, :, 0] = cv2.dilate(image_raw[:, :, 0], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
    image_raw[:, :, 1] = cv2.dilate(image_raw[:, :, 1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))
    image_raw[:, :, 2] = cv2.dilate(image_raw[:, :, 2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius)))

    mask = image_raw.sum(axis=2) > 0
    image[mask] = image_raw[mask]
    overlay[mask] = overlay[mask]

    return image, overlay