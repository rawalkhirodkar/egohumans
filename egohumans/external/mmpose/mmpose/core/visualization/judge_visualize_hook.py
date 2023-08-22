import functools
from mmcv.runner import HOOKS, Hook
import os
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .visualize_hook import batch_unnormalize_image, save_batch_heatmaps, get_max_preds

#####--------------------------------------------------------
@HOOKS.register_module()
class JudgeVisualizeHook(Hook):

    def __init__(self, vis_every_iters=200, max_samples=16, scale=4,):
        self.vis_every_iters = vis_every_iters
        self.max_samples = max_samples
        self.scale = scale
        return

    def before_run(self, runner):
        pass
        return

    def after_run(self, runner):
        pass
        return

    def before_epoch(self, runner):
        pass
        return

    def after_epoch(self, runner):
        pass
        return

    def before_iter(self, runner):
        pass
        return
    
    def after_val_iter(self, runner):
        if runner._inner_iter % self.vis_every_iters != 0:
            return
        
        ## check if the rank is 0
        if not runner.rank == 0:
            return
        
        data_batch = runner.data_batch
        image = data_batch['img'] ## this is normalized
        target = data_batch['target']
        target_weight = data_batch['target_weight']

        ## check if runner.model.judge_keypoint_idx attribute exists
        if not hasattr(runner.model, 'judge_keypoint_idx'):
            judge_keypoint_idx = runner.model.module.judge_keypoint_idx
        else:
            judge_keypoint_idx = runner.model.judge_keypoint_idx

         ## pick the judge keypoint
        target = target[:, judge_keypoint_idx, :, :].unsqueeze(1) ## shape: (batch_size, 1, heatmap_height, heatmap_width)
        target_weight = target_weight[:, judge_keypoint_idx] ## shape: (batch_size, 1)

        outputs = runner.outputs['results']
        output = outputs['output'] ## judge output
        heatmap = outputs['heatmap'] ## predicted heatmap
        gt_classes = outputs['gt_classes'] ## ground truth classes
        gt_dist = outputs['gt_dist'] ## ground truth distance
        pred_classes = outputs['pred_classes'] ## predicted classes
        pred_dist = outputs['pred_dist'] ## predicted distance

        if len(image) > self.max_samples:
            image = image[:self.max_samples]
            target = target[:self.max_samples]
            target_weight = target_weight[:self.max_samples]
            output = output[:self.max_samples]
            heatmap = heatmap[:self.max_samples]
            gt_classes = gt_classes[:self.max_samples]
            gt_dist = gt_dist[:self.max_samples]
            pred_classes = pred_classes[:self.max_samples]
            pred_dist = pred_dist[:self.max_samples]
        
        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'val')
        suffix = str(runner._iter).zfill(6)

        original_image = batch_unnormalize_image(image) ## recolored

        ## concatenate target and heatmap
        target_heatmap = torch.cat([target, heatmap], dim=1)

        save_batch_heatmaps(original_image, target_heatmap, '{}_{}_hm_gt_pred.jpg'.format(prefix, suffix), normalize=False, scale=self.scale)
        self.save_batch_image_judge(255*original_image, target, heatmap, target_weight, \
                                    gt_classes, gt_dist, pred_classes, pred_dist, \
                                    '{}_{}_judge.jpg'.format(prefix, suffix), scale=self.scale)
        return


    def after_iter(self, runner):
        if runner._iter % self.vis_every_iters != 0:
            return

        ## check if the rank is 0
        if not runner.rank == 0:
            return
        
        ## check if runner.model.judge_keypoint_idx attribute exists
        if not hasattr(runner.model, 'judge_keypoint_idx'):
            judge_keypoint_idx = runner.model.module.judge_keypoint_idx
        else:
            judge_keypoint_idx = runner.model.judge_keypoint_idx

        ##------------------------------------
        data_batch = runner.data_batch
        image = data_batch['img'] ## this is normalized
        target = data_batch['target']
        target_weight = data_batch['target_weight']

        ## pick the judge keypoint
        target = target[:, judge_keypoint_idx, :].unsqueeze(1)
        target_weight = target_weight[:, judge_keypoint_idx, :]

        outputs = runner.outputs
        output = outputs['output'] ## judge output
        heatmap = outputs['heatmap'] ## predicted heatmap
        gt_classes = outputs['gt_classes'] ## ground truth classes
        gt_dist = outputs['gt_dist'] ## ground truth distance
        pred_classes = outputs['pred_classes'] ## predicted classes
        pred_dist = outputs['pred_dist'] ## predicted distance

        if len(image) > self.max_samples:
            image = image[:self.max_samples]
            target = target[:self.max_samples]
            target_weight = target_weight[:self.max_samples]
            output = output[:self.max_samples]
            heatmap = heatmap[:self.max_samples]
            gt_classes = gt_classes[:self.max_samples]
            gt_dist = gt_dist[:self.max_samples]
            pred_classes = pred_classes[:self.max_samples]
            pred_dist = pred_dist[:self.max_samples]

        ##------------------------------------
        vis_dir = os.path.join(runner.work_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        prefix = os.path.join(vis_dir, 'train')
        suffix = str(runner._iter).zfill(6)

        original_image = batch_unnormalize_image(image) ## recolored

        ## concatenate target and heatmap
        target_heatmap = torch.cat([target, heatmap], dim=1)

        save_batch_heatmaps(original_image, target_heatmap, '{}_{}_hm_gt_pred.jpg'.format(prefix, suffix), normalize=False, scale=self.scale)
        self.save_batch_image_judge(255*original_image, target, heatmap, target_weight, \
                                    gt_classes, gt_dist, pred_classes, pred_dist, \
                                    '{}_{}_judge.jpg'.format(prefix, suffix), scale=self.scale)

        return
    
    def save_batch_image_judge(self, batch_image, batch_heatmaps, batch_pred_heatmaps, batch_target_weight, \
                                gt_classes, gt_dist, pred_classes, pred_dist, \
                            file_name, scale=4, nrow=8, padding=2):
        '''
        batch_image: [batch_size, channel, height, width]
        batch_joints: [batch_size, num_joints, 3],
        batch_joints_vis: [batch_size, num_joints, 1],
        }
        '''

        B, C, H, W = batch_image.size()

        ## check if type of batch_heatmaps is numpy.ndarray
        if isinstance(batch_heatmaps, np.ndarray):
            batch_joints, _ = get_max_preds(batch_heatmaps)
        else:
            batch_joints, _ = get_max_preds(batch_heatmaps.detach().cpu().numpy())

        ## check if type of batch_heatmaps is numpy.ndarray
        if isinstance(batch_pred_heatmaps, np.ndarray):
            batch_pred_joints, _ = get_max_preds(batch_pred_heatmaps)
        else:
            batch_pred_joints, _ = get_max_preds(batch_pred_heatmaps.detach().cpu().numpy())

        batch_joints = batch_joints*scale ## 4 is the ratio of output heatmap and input image
        batch_pred_joints = batch_pred_joints*scale ## 4 is the ratio of output heatmap and input image

        if isinstance(batch_joints, torch.Tensor):
            batch_joints = batch_joints.cpu().numpy()
        
        if isinstance(batch_pred_joints, torch.Tensor):
            batch_pred_joints = batch_pred_joints.cpu().numpy()

        if isinstance(batch_target_weight, torch.Tensor):
            batch_target_weight = batch_target_weight.cpu().numpy()
            batch_target_weight = batch_target_weight.reshape(B, -1) ## B x 1

        grid = []
        circle_size = 10
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_alpha = 0.5

        for i in range(B):
            image = batch_image[i].permute(1, 2, 0).cpu().numpy() #image_size x image_size x RGB
            image = image.copy()
            kps = batch_joints[i][0] ## size = 2
            pred_kps = batch_pred_joints[i][0] ## size = 2

            ## draw kps as a circle blue color on the image
            kps = kps.astype(np.int32)
            pred_kps = pred_kps.astype(np.int32)

            ## draw kps as a circle bright green color on the image
            kp_vis_image = cv2.circle(image, (kps[0], kps[1]), circle_size, (0, 255, 0), -1) ## as green, the image is RGB
            kp_vis_image = cv2.circle(kp_vis_image, (pred_kps[0], pred_kps[1]), circle_size, (255, 0, 0), -1) ## as red, the image is RGB

            gt_class = gt_classes[i].item()
            pred_class = pred_classes[i].item()

            gt_dist_value = round(gt_dist[i].item(), 2)
            pred_dist_value = round(pred_dist[i].item(), 2)

            ## before the font print, draw a transparent rectangle in gray color on the image
            overlay = kp_vis_image.copy() 
            cv2.rectangle(overlay, (0, 0), (int(kp_vis_image.shape[1]/1.4), 85), (128, 128, 128), -1) ## as gray, the image is RGB

            image_new = cv2.addWeighted(overlay, font_alpha, kp_vis_image, 1 - font_alpha, 0) 

            ## write on the image, orange color in RGB
            image_new = cv2.putText(image_new, 'gt: {}, {}'.format(gt_class, gt_dist_value), (10, 60), font, font_scale, (0, 255, 0), 4, cv2.LINE_AA)
            image_new = cv2.putText(image_new, 'pred: {}, {}'.format(pred_class, pred_dist_value), (10, 25), font, font_scale, (255, 0, 0), 4, cv2.LINE_AA)

            image_new = image_new.transpose((2, 0, 1)).astype(np.float32)
            image_new = torch.from_numpy(image_new.copy())
            grid.append(image_new)

        grid = torchvision.utils.make_grid(grid, nrow, padding)
        ndarr = grid.byte().permute(1, 2, 0).cpu().numpy()
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name, ndarr)

        return
