# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmdet.models import build_detector

from mmtrack.core import outs2results, results2outs
from ..builder import MODELS, build_motion, build_tracker
from .base import BaseMultiObjectTracker
import numpy as np


## returns T x points_3d
def linear_transform(points_3d, T):
    assert(points_3d.shape[1] == 3)

    points_3d_homo = np.ones((4, points_3d.shape[0]))
    points_3d_homo[:3, :] = np.copy(points_3d.T)

    points_3d_prime_homo = np.dot(T, points_3d_homo)
    points_3d_prime = points_3d_prime_homo[:3, :]/ points_3d_prime_homo[3, :]
    points_3d_prime = points_3d_prime.T
    return points_3d_prime


@MODELS.register_module()
class ByteTrackCustom(BaseMultiObjectTracker):
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

    This multi object tracker is the implementation of `ByteTrack
    <https://arxiv.org/abs/2110.06864>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        motion (dict): Configuration of motion. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 detector=None,
                 tracker=None,
                 motion=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if detector is not None:
            self.detector = build_detector(detector)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        return self.detector.forward_train(*args, **kwargs)


    def forward_detector(self, imgs, img_metas, **kwargs):
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_detector_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'

        return None

    def simple_detector_test(self, img, img_metas, rescale=False, **kwargs):

        image_height = img_metas[0]['ori_shape'][0]
        image_width = img_metas[0]['ori_shape'][1]
        image_path = img_metas[0]['filename']

        det_results = self.detector.simple_test(img, img_metas, rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.' ### det_results: B, C, 4 x 5
        assert len(img) == 1

        bbox_results = det_results[0]
        outs_det = results2outs(bbox_results=bbox_results)
        det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        bboxes = outs_det['bboxes'].copy()

        detections = []

        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            score = bbox[4]

            x1 = max(1, x1)
            y1 = max(1, y1)

            x2 = min(x2, image_width)
            y2 = min(y2, image_height)

            detection = np.array([int(x1), int(y1), int(x2), int(y2), score]).reshape(1, -1)
            detections.append(detection)

        if len(detections) != 0:
            detections = np.concatenate(detections, axis=0)
            return detections, image_path

        else:
            return None, None

    def simple_test(self, img, img_metas, rescale=False, public_bboxes=None, roots_3d=None, \
        egoformer_yolox_detections=None, egoformer_yolox_roots_3d=None, **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """

        frame_id = img_metas[0].get('frame_id', -1)
        image_height = img_metas[0]['ori_shape'][0]
        image_width = img_metas[0]['ori_shape'][1]

        if frame_id == 0:
            self.tracker.reset()

        det_results = self.detector.simple_test(img, img_metas, rescale=rescale)
        assert len(det_results) == 1, 'Batch inference is not supported.' ### det_results: B, C, 4 x 5

        if public_bboxes is not None:
            temp = []
            for box in public_bboxes:
                img_scale = img_metas[0].get('scale_factor', 1)
                box = box[0].cpu().numpy()
                box = box / img_scale ## scale
                box = np.concatenate([box, np.ones((len(box), 1))], axis=1)
                temp.append([box])
            det_results = temp            

        bbox_results = det_results[0]
        num_classes = len(bbox_results)

        outs_det = results2outs(bbox_results=bbox_results)
        det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        det_labels = torch.from_numpy(outs_det['labels']).to(img).long()

        ##-----egoformer root3d predictions----------
        if egoformer_yolox_detections is not None:

            if len(outs_det['bboxes']) > 0:
                roots_3d = egoformer_yolox_roots_3d[0]
                temp1 = egoformer_yolox_detections[0][0][0].cpu().numpy()
                temp2 = det_results[0][0]

                ##normalize yolox offline detections
                temp1[:, 0] = np.clip(temp1[:, 0], 0, image_width)
                temp1[:, 1] = np.clip(temp1[:, 1], 0, image_height)
                temp1[:, 2] = np.clip(temp1[:, 2], 0, image_width)
                temp1[:, 3] = np.clip(temp1[:, 3], 0, image_height)

                 ##normalize yolox online detections
                temp2[:, 0] = np.clip(temp2[:, 0], 0, image_width)
                temp2[:, 1] = np.clip(temp2[:, 1], 0, image_height)
                temp2[:, 2] = np.clip(temp2[:, 2], 0, image_width)
                temp2[:, 3] = np.clip(temp2[:, 3], 0, image_height)

                diff = (temp1 - temp2)**2/image_height

                if diff.sum() > 10:
                    print(temp1)
                    print(temp2)
                    print(diff.sum())
                    print(img_metas[0])

                assert(diff.sum() < 10) ## that the yolox bbox match the inputs to the egoformer

            elif len(outs_det['bboxes']) == 0: 
                roots_3d = [[det_bboxes.clone()]] ## dummy dimensions

        ##-------------------------------------------
        if roots_3d is not None:
            assert(len(roots_3d) == 1)
            roots_3d = roots_3d[0][0] ## B x 3

        ## using offshelf detector with Midas regression
        if roots_3d is None and public_bboxes is None:
            extrinsics = img_metas[0]['extrinsics'] ### extrinsics is 4 x 4
            depth = img_metas[0]['depth'] ## depth map, 1408 x 1408
            bboxes = outs_det['bboxes'].copy()

            roots_3d = []

            for bbox_id in range(len(bboxes)):
                bbox = bboxes[bbox_id]
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                score = bbox[4]

                x1 = max(1, x1)
                y1 = max(1, y1)

                x2 = min(x2, image_width)
                y2 = min(y2, image_height)

                bbox_depth = depth[int(y1): int(y2), int(x1): int(x2)]
                camera_avg_depth = bbox_depth.mean()    
                camera_root_3d = np.array([(x1 + x2)/2, (y1 + y2)/2, camera_avg_depth])
                global_root_3d = linear_transform(camera_root_3d.reshape(1, -1), T=extrinsics)[0]
                roots_3d.append(global_root_3d.reshape(1, -1))

            if len(bboxes) != 0:
                roots_3d = np.concatenate(roots_3d, axis=0)
                roots_3d = torch.from_numpy(roots_3d).to(img)
            else:
                roots_3d = det_bboxes.clone()

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            roots_3d=roots_3d,
            **kwargs)

        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])
