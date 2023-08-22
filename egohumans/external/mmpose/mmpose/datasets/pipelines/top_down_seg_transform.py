# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES
from pycocotools import mask as maskUtils
from torchvision.transforms import functional as F

##--------------------------------------------------------------------------------------------------------------------##
COCO_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', \
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', \
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', \
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_TORSO_JOINT_NAMES = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
COCO_TORSO_JOINT_INDEXES = [COCO_JOINT_NAMES.index(joint_name) for joint_name in COCO_TORSO_JOINT_NAMES]

##--------------------------------------------------------------------------------------------------------------------##
@PIPELINES.register_module()
class LoadSeg:
    def __init__(self, num_smooth_iters=5) -> None:
        self.num_smooth_iters = num_smooth_iters
        return

    def __call__(self, results):
        segm = results['segmentation']

        ## if segm is 2D numpy array, then it is already a mask
        ## only used for testing
        if type(segm) == np.ndarray and len(segm.shape) == 2:
            mask = segm
        else:
            h = results['original_height']
            w = results['original_width']
            
            rle = None
            if type(segm) == list:
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, h, w)
                rle = maskUtils.merge(rles)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, h, w)

            assert rle is not None
            mask = maskUtils.decode(rle)
            mask = mask*255        

        ## apply gaussian blur in a for loop 5 times
        for i in range(self.num_smooth_iters):
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        results['segmentation'] = mask
        return results

##--------------------------------------------------------------------------------------------------------------------##
@PIPELINES.register_module()
class TopDownSegRandomDrop:
    def __init__(self, drop_prob=0.5, drop_ratio=0.1) -> None:
        self.drop_prob = drop_prob
        self.drop_ratio = drop_ratio
        return

    def __call__(self, results):
        segmentation = results['segmentation']

        ## randomy zero out drop_ratio pixels of the segmentation mask
        if np.random.rand() <= self.drop_prob:
            mask = np.ones_like(segmentation)
            mask[np.random.rand(*mask.shape) < self.drop_ratio] = 0
            segmentation = segmentation * mask

        results['segmentation'] = segmentation

        return results

##--------------------------------------------------------------------------------------------------------------------##
@PIPELINES.register_module()
class TopDownSegRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']
        segm = results['segmentation']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        flipped = False
        if np.random.rand() <= self.flip_prob:
            flipped = True
            if not isinstance(img, list):
                img = img[:, ::-1, :]
            else:
                img = [i[:, ::-1, :] for i in img]
            if not isinstance(img, list):
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img.shape[1],
                    results['ann_info']['flip_pairs'])
                center[0] = img.shape[1] - center[0] - 1
            else:
                joints_3d, joints_3d_visible = fliplr_joints(
                    joints_3d, joints_3d_visible, img[0].shape[1],
                    results['ann_info']['flip_pairs'])
                center[0] = img[0].shape[1] - center[0] - 1

            if not isinstance(segm, list):
                segm = segm[:, ::-1]
            else:
                segm = [s[:, ::-1] for s in segm]

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center
        results['flipped'] = flipped
        results['segmentation'] = segm

        return results

##--------------------------------------------------------------------------------------------------------------------##
@PIPELINES.register_module()
class TopDownSegAffine:
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified keys:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']
        segm = results['segmentation']

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
                
                segm = cv2.warpAffine(
                    segm,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)

            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
                segm = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in segm
                ]

            joints_3d[:, 0:2] = \
                warp_affine_joints(joints_3d[:, 0:2].copy(), trans)

        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
                
                segm = cv2.warpAffine(
                    segm,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)

            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
                
                segm = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in segm
                ]

            for i in range(results['ann_info']['num_joints']):
                if joints_3d_visible[i, 0] > 0.0:
                    joints_3d[i,
                              0:2] = affine_transform(joints_3d[i, 0:2], trans)

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['segmentation'] = segm

        return results

@PIPELINES.register_module()
class ToTensorSeg:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        if isinstance(results['img'], (list, tuple)):
            results['img'] = [F.to_tensor(img) for img in results['img']]
        else:
            results['img'] = F.to_tensor(results['img'])
        
        results['segmentation'] = F.to_tensor(results['segmentation'])

        return results