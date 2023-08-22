# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
from mmcv.image import imwrite
from mmcv.utils.misc import deprecated_api_warning
from mmcv.visualization.image import imshow

from torch import nn
from torch.nn import functional as F

from mmpose.core import imshow_bboxes, imshow_keypoints
from .. import builder
from ..builder import POSENETS
from .base import BasePose
from mmpose.models import build_posenet
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch
from mmpose.core.evaluation.top_down_eval import _get_max_preds

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

@POSENETS.register_module()
class JudgeTopDown(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
    """

    def __init__(self,
                 backbone,
                 judge_keypoint_idx,
                 jitter_prob,
                 pose_model_dict,
                 pose_model_checkpoint,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__()
        self.fp16_enabled = False

        self.judge_keypoint_idx = judge_keypoint_idx
        
        self.pose_model = build_posenet(pose_model_dict)
        load_checkpoint(self.pose_model, pose_model_checkpoint, map_location='cpu')

        ## freeze the pose model
        for param in self.pose_model.parameters():
            param.requires_grad = False

        self.backbone = builder.build_backbone(backbone)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = backbone['num_classes']
        
        # # Use KL-Divergence Loss
        # self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        # self.criterion = FocalLoss()

        ## use cross entropy loss
        self.criterion = torch.nn.CrossEntropyLoss()

        # Create bin edges
        # Since log(1) is 0, we start from a very small number (close to zero) and end at 0
        # self.bin_edges = torch.tensor(np.logspace(-2, 0, self.num_classes + 1))  # We add 1 because for n bins, we need n+1 edges

        ## use linear bin edges
        self.bin_edges = torch.linspace(0, 1, self.num_classes + 1)

        self.bin_edges[0] = 0.0  # Set the first edge to 0
        self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2 ## shape: (num_classes, )

        self.image_size = backbone['img_size']
        self.max_distance = np.sqrt((self.image_size[0]/4)**2 + (self.image_size[1]/4)**2) ## 140 pixel for 448x336 image

        self.bin_distances = self.bin_centers * self.max_distance ## shape: (num_classes, )

        self.jitter_prob = jitter_prob
        self.pixel_offset = int(self.max_distance * 0.75)

        self.init_weights(pretrained=pretrained)
        return

    @property
    def with_neck(self):
        """Check if has neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, target, target_weight, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        with torch.no_grad():
            result = self.pose_model(img=img, img_metas=img_metas, return_loss=False, return_heatmap=True, **kwargs) ## this will do left and right flip and return the average heatmap
            heatmap = result['output_heatmap'] ## is a numpy array
            heatmap = torch.from_numpy(heatmap).to(img.device)

        ## select the judge keypoint
        heatmap = heatmap[:, self.judge_keypoint_idx, :, :].unsqueeze(1) ## shape: (batch_size, 1, heatmap_height, heatmap_width)
        target = target[:, self.judge_keypoint_idx, :, :].unsqueeze(1) ## shape: (batch_size, 1, heatmap_height, heatmap_width)
        target_weight = target_weight[:, self.judge_keypoint_idx] ## shape: (batch_size, 1)

        if np.random.rand() < self.jitter_prob:
            batch_size = heatmap.shape[0]
            for i in range(batch_size):
                # Generate random shifts for x and y direction
                shifts_y = np.random.randint(-self.pixel_offset, self.pixel_offset + 1)
                shifts_x = np.random.randint(-self.pixel_offset, self.pixel_offset + 1)

                # Apply the shifts to each sample individually
                heatmap[i] = torch.roll(heatmap[i], shifts=(shifts_y, shifts_x), dims=(1, 2))

        ## convert target heatmap to target class
        gt_loc, _ = _get_max_preds(target.cpu().numpy())
        gt_loc = gt_loc.reshape(-1, 2)
        gt_loc = torch.from_numpy(gt_loc).to(img.device) ## B x 2
        gt_loc = gt_loc * target_weight.repeat(1, 2) ## B x 2

        pred_loc, _ = _get_max_preds(heatmap.cpu().numpy())
        pred_loc = pred_loc.reshape(-1, 2)
        pred_loc = torch.from_numpy(pred_loc).to(img.device)
        pred_loc = pred_loc * target_weight.repeat(1, 2) ## B x 2

        ## compute L2 euclidean distance between gt and pred, mask by target_weight
        gt_dist = torch.norm(gt_loc - pred_loc, dim=1) ## shape: (batch_size, 1)   
        gt_classes = self.class_from_distance(gt_dist, target_weight) ## shape: (batch_size, 1)

        output = self.backbone(img, heatmap) # it is a classifier, return logits

        pred_classes = torch.argmax(output, dim=1) ## shape: (batch_size, 1)
        pred_dist = self.bin_distances[pred_classes] ## shape: (batch_size, 1)

        # ## plot bin edges
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(self.bin_edges.cpu().numpy(), np.arange(self.num_classes + 1))
        # plt.show()

        # if return loss
        losses = dict()
        classification_losses = self.get_loss(output, gt_classes, target_weight.view(-1))
        losses.update(classification_losses)

        accuracy = self.get_accuracy(output, gt_classes, target_weight.view(-1))
        losses.update(accuracy)

        return losses, output, heatmap, gt_classes, gt_dist, pred_classes, pred_dist
    
    def forward_test(self, img, img_metas, target, target_weight, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)
        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        with torch.no_grad():
            result = self.pose_model(img=img, img_metas=img_metas, return_loss=False, return_heatmap=True, **kwargs) ## this will do left and right flip and return the average heatmap
            heatmap = result['output_heatmap'] ## is a numpy array
            heatmap = torch.from_numpy(heatmap).to(img.device)
        
        ## select the judge keypoint
        heatmap = heatmap[:, self.judge_keypoint_idx, :, :].unsqueeze(1) ## shape: (batch_size, 1, heatmap_height, heatmap_width)
        target = target[:, self.judge_keypoint_idx, :, :].unsqueeze(1) ## shape: (batch_size, 1, heatmap_height, heatmap_width)
        target_weight = target_weight[:, self.judge_keypoint_idx] ## shape: (batch_size, 1)

        ## convert target heatmap to target class
        gt_loc, _ = _get_max_preds(target.cpu().numpy())
        gt_loc = gt_loc.reshape(-1, 2)
        gt_loc = torch.from_numpy(gt_loc).to(img.device) ## B x 2
        gt_loc = gt_loc * target_weight.repeat(1, 2) ## B x 2

        pred_loc, _ = _get_max_preds(heatmap.cpu().numpy())
        pred_loc = pred_loc.reshape(-1, 2)
        pred_loc = torch.from_numpy(pred_loc).to(img.device)
        pred_loc = pred_loc * target_weight.repeat(1, 2) ## B x 2

        ## compute L2 euclidean distance between gt and pred, mask by target_weight
        gt_dist = torch.norm(gt_loc - pred_loc, dim=1) ## shape: (batch_size, 1)   
        gt_classes = self.class_from_distance(gt_dist, target_weight) ## shape: (batch_size, 1)

        ##------------------------------------
        result = {}

        output = self.backbone(img, heatmap) # it is a classifier
        pred_classes = torch.argmax(output, dim=1) ## shape: (batch_size, 1)
        pred_dist = self.bin_distances[pred_classes] ## shape: (batch_size, 1)
        
        ##------------------------------------
        result['output'] = output.detach().cpu().clone()
        result['heatmap'] = heatmap.detach().cpu().clone()
        result['gt_classes'] = gt_classes.detach().cpu().clone()
        result['gt_dist'] = gt_dist.detach().cpu().clone()
        result['pred_classes'] = pred_classes.detach().cpu().clone()
        result['pred_dist'] = pred_dist.detach().cpu().clone()

        return result

    def get_loss(self, output, gt_classes, target_weight,):
        # Exclude the indexes where target_weight is 0 (not part of the target)
        valid_indexes = torch.where(target_weight > 0)

        if valid_indexes[0].shape[0] == 0:
            loss = 0*output.sum()
            # If no valid indexes, return zero loss
            return {'classification_loss': loss}

        output_valid = output[valid_indexes]
        gt_classes_valid = gt_classes[valid_indexes]

        ## return cross entropy loss
        loss = self.criterion(output_valid, gt_classes_valid.long())

        # Returning it as a dict
        return {'classification_loss': loss}


    def get_loss_v0(self, output, gt_classes, target_weight, window=3):
        # Exclude the indexes where target_weight is 0 (not part of the target)
        valid_indexes = torch.where(target_weight > 0)

        if valid_indexes[0].shape[0] == 0:
            loss = 0*output.sum()
            # If no valid indexes, return zero loss
            return {'classification_loss': loss}

        output_valid = output[valid_indexes]
        gt_classes_valid = gt_classes[valid_indexes]

        # Create soft labels using Gaussian smoothing
        gt_classes_valid_float = gt_classes_valid.float().unsqueeze(1)  # Add a dimension to allow broadcasting
        soft_labels = self.gaussian_smooth_label(gt_classes_valid_float, window)

        # The loss is only computed on valid indexes (where target_weight > 0)
        loss = self.criterion(output_valid, soft_labels)

        # Returning it as a dict
        return {'classification_loss': loss}

    def gaussian_smooth_label(self, label, window):
        """Apply gaussian smoothing on labels."""
        labels_range = torch.arange(self.num_classes).float().to(label.device)
        label_float = label.float().unsqueeze(1)  # Add a dimension to allow broadcasting
        
        # Create gaussian weights
        gauss_label = torch.exp(-torch.pow(labels_range - label_float, 2) / (2 * window * window))

        ## zero out if the label is outside the window
        gauss_label = torch.where(torch.abs(labels_range - label_float) > window*window, torch.tensor(0.0).to(label.device), gauss_label)

        ## remove the fake dimension
        gauss_label = gauss_label.squeeze(1)

        return gauss_label
    
    ## convert distance to a class index    
    def class_from_distance(self, dist, target_weight):
        target_weight = target_weight.view(-1)

        ## clamp the distance to be within 0 and max_distance
        dist = torch.clamp(dist, 0 + 1e-6, self.max_distance - 1e-6) ## shape: (batch_size, 1)

        # Compute the raw score
        score = dist / self.max_distance ## shape: (batch_size, 1)

        # Use digitize to assign each score to a bin
        score_classes = torch.bucketize(score, self.bin_edges.to(dist.device)) 

         # When target_weight is 0, set score_classes to -1
        score_classes = torch.where(target_weight == 0, torch.tensor(-1).to(dist.device), score_classes) - 1

        return score_classes

    def get_accuracy(self, output, gt_classes, target_weight):
        # Exclude the indexes where target_weight is 0 (not part of the target)
        valid_indexes = torch.where(target_weight > 0)

        if valid_indexes[0].shape[0] == 0:
            # If no valid indexes, return zero accuracy
            return {'accuracy': 0.0}

        output_valid = output[valid_indexes]
        gt_classes_valid = gt_classes[valid_indexes]

        # Use softmax for normalization
        probabilities = torch.nn.functional.softmax(output_valid, dim=1)

        # Get class indices from probabilities
        pred_classes = torch.argmax(probabilities, dim=1)

        # Compute accuracy
        correct_predictions = torch.sum(pred_classes == gt_classes_valid).float()
        total_predictions = gt_classes_valid.shape[0]
        accuracy = correct_predictions / total_predictions

        # Returning it as a dict
        return {'accuracy': accuracy}

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Output heatmaps.
        """
        with torch.no_grad():
            result = self.pose_model(img) ## this will do left and right flip and return the average heatmap
            heatmap = result['output_heatmap'] ## is a numpy array
            heatmap = torch.from_numpy(heatmap).to(img.device)

        output = self.backbone(img, heatmap) # it is a classifier
        return output

    @deprecated_api_warning({'pose_limb_color': 'pose_link_color'},
                            cls_name='TopDown')
    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_link_color=None,
                    text_color='white',
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    bbox_thickness=1,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()

        bbox_result = []
        bbox_labels = []
        pose_result = []
        for res in result:
            if 'bbox' in res:
                bbox_result.append(res['bbox'])
                bbox_labels.append(res.get('label', None))
            pose_result.append(res['keypoints'])

        if bbox_result:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            imshow_bboxes(
                img,
                bboxes,
                labels=bbox_labels,
                colors=bbox_color,
                text_color=text_color,
                thickness=bbox_thickness,
                font_scale=font_scale,
                show=False)

        if pose_result:
            imshow_keypoints(img, pose_result, skeleton, kpt_score_thr,
                             pose_kpt_color, pose_link_color, radius,
                             thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses, output, heatmap, gt_classes, gt_dist, pred_classes, pred_dist = self.forward(**data_batch)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))),
            output=output.detach().cpu().clone(),
            heatmap=heatmap.detach().cpu().clone(),
            gt_classes=gt_classes.detach().cpu().clone(),
            gt_dist=gt_dist.detach().cpu().clone(),
            pred_classes=pred_classes.detach().cpu().clone(),
            pred_dist=pred_dist.detach().cpu().clone(),
            )

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        results = self.forward(return_loss=False, return_heatmap=True, **data_batch) ## overloaded from base to return heatmap

        outputs = dict(results=results)

        return outputs