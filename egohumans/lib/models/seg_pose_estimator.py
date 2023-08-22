import numpy as np
import os
import cv2

## this is importing from vitpose and not the mmpose repo
from mmpose.apis import (inference_top_down_seg_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from tqdm import tqdm
from mmdet.apis import inference_detector, init_detector
from mmpose.core.bbox.transforms import bbox_xyxy2xywh, bbox_xywh2cs, bbox_cs2xywh, bbox_xywh2xyxy
from .segmentator import SegmentationModel

##------------------------------------------------------------------------------------
class SegPoseModel:
    def __init__(self, cfg, pose_config=None, pose_checkpoint=None):
        self.cfg = cfg
        self.pose_config = pose_config
        self.pose_checkpoint = pose_checkpoint

        ## load pose model
        self.pose_model = init_pose_model(self.pose_config, self.pose_checkpoint, device='cuda:0'.lower())
        self.dataset = self.pose_model.cfg.data['test']['type']
        self.dataset_info = DatasetInfo(self.pose_model.cfg.data['test'].get('dataset_info', None))
        self.return_heatmap = False
        self.output_layer_names = None
        self.num_keypoints =  len(self.pose_model.cfg.dataset_info['keypoint_info'].keys())
        self.coco_17_keypoints_idxs = np.array(range(17)) ## indexes of the COCO keypoints
        self.coco_17_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

        ##------hyperparameters-----
        self.bbox_thres = self.cfg.POSE2D.BBOX_THRES ## Bounding box score threshold
        self.rgb_keypoint_thres = self.cfg.POSE2D.RGB_THRES ## Keypoint score threshold
        self.gray_keypoint_thres = self.cfg.POSE2D.GRAY_THRES ## Keypoint score threshold

        self.rgb_keypoint_vis_thres = self.cfg.POSE2D.RGB_VIS_THRES ## Keypoint score threshold
        self.gray_keypoint_vis_thres = self.cfg.POSE2D.GRAY_VIS_THRES ## Keypoint score threshold

        ## Keypoint radius for visualization
        self.radius = { \
                        ('exo', 'rgb'): self.cfg.POSE2D.VIS.RADIUS.EXO_RGB,
                        ('ego', 'rgb'): self.cfg.POSE2D.VIS.RADIUS.EGO_RGB,
                        ('ego', 'left'): self.cfg.POSE2D.VIS.RADIUS.EGO_LEFT,
                        ('ego', 'right'): self.cfg.POSE2D.VIS.RADIUS.EGO_RIGHT,
                    } 

        ## Link thickness for visualization
        self.thickness = { \
                        ('exo', 'rgb'): self.cfg.POSE2D.VIS.THICKNESS.EXO_RGB,
                        ('ego', 'rgb'): self.cfg.POSE2D.VIS.THICKNESS.EGO_RGB,
                        ('ego', 'left'): self.cfg.POSE2D.VIS.THICKNESS.EGO_LEFT,
                        ('ego', 'right'): self.cfg.POSE2D.VIS.THICKNESS.EGO_RIGHT,
                    } 

        self.min_vis_keypoints = self.cfg.POSE2D.MIN_VIS_KEYPOINTS ## coco format, 17 keypoints!
        self.kps_oks_thres = self.cfg.POSE2D.OVERLAP_OKS_THRES

        return 

    ####--------------------------------------------------------
    def get_poses2d(self, segmentations, image_name, camera_type='ego', camera_mode='rgb', camera=None, aria_humans=None,):

        ## reformat the segmentations to the format required by mmpose
        bboxes = []
        for human_name in segmentations.keys():
            segmentation = segmentations[human_name]
            segmentation['human_name'] = human_name

            ## convert mask boolean array to float array
            if segmentation['segmentation'] is not None:
                mask = segmentation['segmentation'].astype(np.float32)
                segmentation['segmentation'] = mask

                if len(segmentation['bbox']) == 4:
                    ## add 1 at the end for the bbox
                
                    segmentation['bbox'] = np.append(segmentation['bbox'], 1)

                bboxes.append(segmentation)

        pose_results, _ = inference_top_down_seg_pose_model(
                    self.pose_model,
                    image_name,
                    bboxes,
                    bbox_thr=self.bbox_thres,
                    format='xyxy',
                    dataset=self.dataset,
                    dataset_info=self.dataset_info,
                    return_heatmap=self.return_heatmap,
                    outputs=self.output_layer_names)

        ###----remove the poses caused by overlapping humans-----------
        pose_results = self.refine_poses(pose_results, camera, aria_humans, debug=self.cfg.POSE2D.DEBUG)

        ##---------refine the bboxes-------------------
        pose_results = self.refine_bboxes(pose_results, camera_type, camera_mode)

        if len(pose_results) < len(bboxes):
            pose_human_names = [val['human_name'] for val in pose_results] ## human names in the pose results

            for bbox in bboxes:
                if bbox['human_name'] not in pose_human_names:
                    pose_result = bbox.copy()
                    pose_result['keypoints'] = np.zeros((self.num_keypoints, 3)) ## dummy pose
                    pose_results.append(pose_result)

        return pose_results

    ####--------------------------------------------------------
    def refine_poses(self, pose_results, camera, aria_humans, debug=True):
        valid_pose_results = []

        ### remove detections with very few keypoints
        for i in range(len(pose_results)):
            pose = pose_results[i]['keypoints']

            if camera.type_string == 'rgb':
                is_valid = pose[self.coco_17_keypoints_idxs, 2] > self.rgb_keypoint_thres
            else:
                is_valid = pose[self.coco_17_keypoints_idxs, 2] > self.gray_keypoint_thres

            ## skip if low confidence detection
            if is_valid.sum() <= self.min_vis_keypoints:
                continue

            valid_pose_results.append(pose_results[i])

        pose_results = valid_pose_results
        
        ### now do the oks nms
        is_valid_pose = [True]*len(pose_results) ## start with everything valid

        camera_location = camera.location

        for i in range(len(pose_results)):
            detection_i = pose_results[i]
            keypoints_i = detection_i['keypoints'][self.coco_17_keypoints_idxs, :] ## coco 17 keypoints
            human_name_i = detection_i['human_name']
            location_i = aria_humans[human_name_i].location
            dist_to_camera_i = np.sqrt(((location_i - camera_location)**2).sum())

            for j in range(i+1, len(pose_results)):

                if is_valid_pose[i] == False or is_valid_pose[j] == False:
                    continue

                detection_j = pose_results[j]
                keypoints_j = detection_j['keypoints'][self.coco_17_keypoints_idxs, :] ## coco 17 keypoints
                human_name_j = detection_j['human_name']
                location_j = aria_humans[human_name_j].location
                dist_to_camera_j = np.sqrt(((location_j - camera_location)**2).sum())

                kps_oks = self.compute_kps_oks(keypoints_i, keypoints_j, camera)

                if debug == True:
                    print('cam:{}, {}, {}, oks:{}'.format(camera.camera_name, human_name_i, human_name_j, kps_oks))

                if kps_oks > self.kps_oks_thres:

                    if debug == True:
                        print('removing  cam:{}, {}, {}, oks:{}'.format(camera.camera_name, human_name_i, human_name_j, kps_oks))
                        print()

                    # ##----remove both poses-----
                    is_valid_pose[i] = False 
                    is_valid_pose[j] = False 

        valid_pose_results = []

        for i in range(len(is_valid_pose)):
            raw_conf = pose_results[i]['keypoints'][:, 2].copy()
            pose_results[i]['raw_keypoints_confidence'] = raw_conf
            pose_results[i]['is_valid'] = is_valid_pose[i]
            valid_pose_results.append(pose_results[i])

            if is_valid_pose[i] == False:
                pose_results[i]['keypoints'][:, 2] = 0

        return valid_pose_results

    ##-----------------------------------------
    ## object keypoint similarity
    def compute_kps_oks(self, keypoints1, keypoints2, camera):

        if camera.type_string == 'rgb':
            keypoint_thres = self.rgb_keypoint_thres
        else:
            keypoint_thres = self.gray_keypoint_thres

        is_valid = (keypoints1[:, 2] > keypoint_thres) * (keypoints2[:, 2] > keypoint_thres)

        ## probably different skeletons
        if is_valid.sum() == 0:
            return 0.1

        ###------------area------------------
        area1 = self.get_area_from_pose(pose=keypoints1, keypoint_thres=keypoint_thres)
        area2 = self.get_area_from_pose(pose=keypoints2, keypoint_thres=keypoint_thres)

        area = (area1 + area2)/2 ## average the bbox area

        ###------------compute oks------------------
        xg = keypoints1[:, 0]; yg = keypoints1[:, 1]
        xd = keypoints2[:, 0]; yd = keypoints2[:, 1]
        dx = xd - xg
        dy = yd - yg

        vars = (self.coco_17_sigmas * 2)**2
        e = (dx**2 + dy**2) / vars / (area+np.spacing(1)) / 2

        e = e[is_valid > 0]
        oks = np.sum(np.exp(-e)) / e.shape[0]

        return oks

    ####--------------------------------------------------------
    def get_area_from_pose(self, pose, keypoint_thres):
        assert(len(pose) == len(self.coco_17_keypoints_idxs))
        is_valid = pose[:, 2] > keypoint_thres
        
        x1 = pose[is_valid, 0].min(); x2 = pose[is_valid, 0].max()
        y1 = pose[is_valid, 1].min(); y2 = pose[is_valid, 1].max()

        area = (x2-x1)*(y2-y1) ## area of the bbox

        return area
    
    ####--------------------------------------------------------
    def refine_bboxes(self, pose_results, camera_type, camera_mode, padding=1.2, aspect_ratio=3/4):
        valid_pose_results = []

        for i in range(len(pose_results)):
            bbox = pose_results[i]['bbox']
            pose = pose_results[i]['keypoints']

            ## no change if the pose is removed due to oks threshold
            if pose_results[i]['is_valid'] == False:
                valid_pose_results.append(pose_results[i])
                continue

            if camera_mode == 'rgb':
                is_valid = pose[:, 2] > self.rgb_keypoint_thres
            else:
                is_valid = pose[:, 2] > self.gray_keypoint_thres

            x1 = pose[is_valid, 0].min(); x2 = pose[is_valid, 0].max()
            y1 = pose[is_valid, 1].min(); y2 = pose[is_valid, 1].max()

            bbox_xyxy = np.array([[x1, y1, x2, y2]])

            # https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/bbox/transforms.py
            bbox_center, bbox_scale = bbox_xywh2cs(\
                            bbox=bbox_xyxy2xywh(bbox_xyxy).reshape(-1), \
                            aspect_ratio=aspect_ratio, \
                            padding=padding) ## aspect_ratio is w/h

            bbox_xywh = bbox_cs2xywh(center=bbox_center, scale=bbox_scale, padding=1.0)

            refined_bbox = bbox_xywh2xyxy(bbox_xywh.reshape(1, 4)).reshape(-1) ## (4,) ## tight fitting bbox to the detected pose

            ## if exo camera, completely replace the bbox
            if camera_type == 'exo':
                # pose_results[i]['bbox'][:4] = refined_bbox.astype(np.int)
                pose_results[i]['bbox'][:4] = refined_bbox.astype(int)

            ## if ego, take a union of the two rectangles
            if camera_type == 'ego':
                refined_bbox = self.merge_bboxes(bbox[:4], refined_bbox)
                # pose_results[i]['bbox'][:4] = refined_bbox.astype(np.int)
                pose_results[i]['bbox'][:4] = refined_bbox.astype(int)

            valid_pose_results.append(pose_results[i])

        return valid_pose_results

    def merge_bboxes(self, primary_bbox, secondary_bbox):
        x1 = min(primary_bbox[0], secondary_bbox[0])
        y1 = min(primary_bbox[1], secondary_bbox[1])

        x2 = max(primary_bbox[2], secondary_bbox[2])
        y2 = max(primary_bbox[3], secondary_bbox[3])

        bbox = np.array([x1, y1, x2, y2])
        return bbox

    ####--------------------------------------------------------
    def draw_poses2d(self, pose_results, image_name, save_path, camera_type='ego', camera_mode='rgb'):
        bbox_bgr_colors = [result['color'] for result in pose_results]

        if camera_mode == 'rgb':
            keypoint_thres = self.rgb_keypoint_vis_thres
        else:
            keypoint_thres = self.gray_keypoint_vis_thres
        
        bbox_colors = []
        for bgr_color in bbox_bgr_colors:
            bbox_colors.append((bgr_color[0], bgr_color[1], bgr_color[2])) ## bgr colors

        vis_pose_result(
            self.pose_model,
            image_name,
            pose_results,
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            kpt_score_thr=keypoint_thres,
            radius=self.radius[(camera_type, camera_mode)],
            thickness=self.thickness[(camera_type, camera_mode)],
            bbox_color=bbox_colors,
            bbox_thickness=self.thickness[(camera_type, camera_mode)],
            show=False,
            out_file=save_path)

        return

    ####--------------------------------------------------------
    def draw_projected_poses3d(self, pose_results, image_name, save_path, camera_type='ego', camera_mode='rgb'):

        if camera_mode == 'rgb':
            keypoint_thres = self.rgb_keypoint_thres
        else:
            keypoint_thres = self.gray_keypoint_thres

        ##-----------restructure to the desired format used by mmpose---------
        pose_results_ = []
        for human_name in pose_results.keys():
            pose = pose_results[human_name] ## 17 x 3
            pose_ = np.zeros((self.num_keypoints, 3)) ## 133 x 3

            pose_[:len(pose), :3] = pose[:, :]

            pose_result = {'keypoints':pose_}
            pose_results_.append(pose_result)
        
        pose_results = pose_results_            

        vis_pose_result(
            self.pose_model,
            image_name,
            pose_results,
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            kpt_score_thr=keypoint_thres,
            radius=self.radius[(camera_type, camera_mode)],
            thickness=self.thickness[(camera_type, camera_mode)],
            show=False,
            out_file=save_path)

        return