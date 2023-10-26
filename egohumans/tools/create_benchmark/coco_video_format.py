import _init_paths
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import cv2
import matplotlib.pyplot as plt
from cycler import cycle
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import mmcv
from datetime import datetime
from datasets.aria_camera import AriaCamera
from datasets.aria_human import AriaHuman
from datasets.ego_exo_scene import EgoExoScene

from configs import cfg
from configs import update_config
from pycocotools.coco import COCO
from pycococreatortools import pycococreatortools
import pickle 
from utils.transforms import linear_transform

#------------------------------------------------------------------------------------
CATEGORIES = [
    {
        "id": 0,
        "name": "pedestrian"
    }
    
]

COCO_EXO_OUTPUT = {
        "categories": CATEGORIES,
        "videos": [],
        "images": [],
        "annotations": [],
        "video_num": 0,
        "video_image_num": {},
        "video_instance_num": {},
        "image_num": 0,
        "instance_num": 0,
    }

COCO_EXO_DETECTIONS = {'det_bboxes': {}, 'keypoints_3d':{}, 'roots_3d':{}} ## image_path: bbox


COCO_EGO_RGB_OUTPUT = {
        "categories": CATEGORIES,
        "videos": [],
        "images": [],
        "annotations": [],
        "video_num": 0,
        "video_image_num": {},
        "video_instance_num": {},
        "image_num": 0,
        "instance_num": 0,
    }
COCO_EGO_RGB_DETECTIONS = {'det_bboxes': {}, 'keypoints_3d':{}, 'roots_3d':{}} ## image_path: bbox


COCO_EGO_SLAM_OUTPUT = {
        "categories": CATEGORIES,
        "videos": [],
        "images": [],
        "annotations": [],
        "video_num": 0,
        "video_image_num": {},
        "video_instance_num": {},
        "image_num": 0,
        "instance_num": 0,
    }
COCO_EGO_SLAM_DETECTIONS = {'det_bboxes': {}, 'keypoints_3d':{}, 'roots_3d':{}} ## image_path: bbox

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

##------------------------------------------------------------------------------------
def pose_to_bbox(pose, image_width, image_height, keypoint_thres=0.5, padding=1.4, min_keypoints=5):
    is_valid = (pose[:, 2] > keypoint_thres)  
    is_valid = is_valid * (pose[:, 0] > 0) * (pose[:, 0] <= image_width)
    is_valid = is_valid * (pose[:, 1] > 0) * (pose[:, 1] <= image_height)

    if is_valid.sum() < min_keypoints:
        return None, None, None, None, None

    x1 = pose[is_valid, 0].min(); x2 = pose[is_valid, 0].max()
    y1 = pose[is_valid, 1].min(); y2 = pose[is_valid, 1].max()

    center_x = (x1 + x2)/2
    center_y = (y1 + y2)/2

    scale_x = (x2 - x1)*padding
    scale_y = (y2 - y1)*padding

    bbx = max(1, center_x - scale_x/2)
    bby = max(1, center_y - scale_y/2)

    bbw = scale_x
    bbh = scale_y

    return bbx, bby, bbw, bbh, is_valid

def coco_add_to(sequence_name, parent_sequence_name, COCO_OUTPUT, COCO_DETECTIONS, poses2d, poses3d, image_path, image_width, image_height, camera_name, camera_mode, time_stamp, \
        depth=None, extrinsics=None, depth_path=None):
    image_size = (image_width, image_height)

    video_name = '{}_{}_{}_{}'.format(parent_sequence_name, sequence_name, camera_name, camera_mode)
    existing_video_names = [val['name'] for val in COCO_OUTPUT['videos']]

    if video_name not in existing_video_names:
        video_id = COCO_OUTPUT['video_num'] + 1
        video_info = {"id": video_id, "name": video_name}
        COCO_OUTPUT['video_num'] += 1
        COCO_OUTPUT['videos'].append(video_info)
        COCO_OUTPUT['video_image_num'][video_id] = 0
        COCO_OUTPUT['video_instance_num'][video_id] = 0

    video_id = -1
    image_num = -1
    instance_num = -1

    for video_info in COCO_OUTPUT['videos']:
        if video_name == video_info['name']:
            video_id = video_info['id']
            image_num = COCO_OUTPUT['video_image_num'][video_id]
            instance_num = COCO_OUTPUT['video_instance_num'][video_id]
            break

    assert(video_id != -1)
    assert(image_num != -1)
    assert(instance_num != -1)

    ## only for the non aria cameras
    image_info = {
                'file_name': image_path,
                'height': image_height,
                'width': image_width,
                'id': COCO_OUTPUT['image_num'],
                'video_id': video_id,
                "frame_id": image_num,
    }

    if depth is not None:
        assert depth_path is not None
        extrinsics_flat = extrinsics.reshape(-1)
        image_info['extrinsics'] = list(extrinsics_flat)
        image_info['depth_path'] = depth_path

    annotation_info_list = []
    gt_detections = []
    gt_keypoints_3d = []
    gt_roots_3d = []

    for human_name in poses2d.keys():
        bbx, bby, bbw, bbh, is_valid = pose_to_bbox(poses2d[human_name], image_width, image_height)

        ## out of the field of view
        if bbx == None:
            continue

        annotation_info = {'id': COCO_OUTPUT['instance_num'] + len(annotation_info_list), \
                                'image_id': COCO_OUTPUT['image_num'], 
                                'video_id': video_id,
                                'category_id': 0,
                                'instance_id': int(human_name.replace('aria', '')), ##aria01 -> 1, aria02 -> 2
                                'bbox': [int(bbx), int(bby), int(bbw), int(bbh)],
                                'area': int(bbw*bbh),
                                'occluded': False,
                                'truncated': False,
                                'iscrowd': False,
                                'ignore': False,
                                'is_vid_train_frame': True,
                                'visibility': 1.0,
                            }

        kps_v = np.zeros(17)
        kps_v[is_valid] = 2

        ## in coco format
        kps = [0]*17*3

        kps[0::3] = poses2d[human_name][:, 0].round().astype(int)
        kps[1::3] = poses2d[human_name][:, 1].round().astype(int)
        kps[2::3] = kps_v.tolist()

        # annotation_info['keypoints'] = kps
        # annotation_info['num_keypoints'] = is_valid.sum()
        annotation_info['human_name'] = human_name
        annotation_info['camera_name'] = camera_name
        annotation_info['camera_mode'] = camera_mode
        annotation_info['time_stamp'] = time_stamp

        if depth is not None:
            annotation_bbox_depth = depth[int(bby): int(bby+bbh), int(bbx): int(bbx+bbw)]
            camera_avg_depth = annotation_bbox_depth.mean()

            camera_root_3d = np.array([bbx + bbw/2, bby + bbh/2, camera_avg_depth])
            global_root_3d = linear_transform(camera_root_3d.reshape(1, -1), T=extrinsics)[0]
            gt_roots_3d.append(global_root_3d.reshape(1, -1))

        annotation_info_list.append(annotation_info)

        gt_detection = np.array([int(bbx), int(bby), int(bbx + bbw), int(bby + bbh), 1.0]).reshape(1, -1)
        gt_detections.append(gt_detection)

        gt_keypoints_3d.append(poses3d[human_name].reshape(1, -1, 4))

    ## early exit
    if len(annotation_info_list) == 0:
        return

    gt_detections = np.concatenate(gt_detections, axis=0)
    COCO_DETECTIONS['det_bboxes'][image_path] = [gt_detections]

    gt_keypoints_3d = np.concatenate(gt_keypoints_3d, axis=0)
    COCO_DETECTIONS['keypoints_3d'][image_path] = [gt_keypoints_3d]

    if depth is not None:
        gt_roots_3d = np.concatenate(gt_roots_3d, axis=0)
        COCO_DETECTIONS['roots_3d'][image_path] = [gt_roots_3d]

    COCO_OUTPUT['video_image_num'][video_id] += 1
    COCO_OUTPUT["images"].append(image_info)

    COCO_OUTPUT['video_instance_num'][video_id] += len(annotation_info_list)
    COCO_OUTPUT['annotations'] += annotation_info_list

    COCO_OUTPUT['image_num'] += 1
    COCO_OUTPUT['instance_num'] += len(annotation_info_list)

    return 

##------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Visualization of extrinsics of camera parameters.')
    parser.add_argument('--sequence_path', action='store', help='the path to the sequence for visualization')
    parser.add_argument('--output_path', action='store', help='the path to the sequence for visualization')

    args = parser.parse_args()
    sequence_path = args.sequence_path
    sequence_name = sequence_path.split('/')[-1]
    parent_sequence = sequence_name.split('_')[-1]
    config_file = os.path.join(_init_paths.root_path, 'configs', parent_sequence, '{}.yaml'.format(sequence_name))
    update_config(cfg, config_file)

    scene = EgoExoScene(cfg=cfg, root_dir=sequence_path)
    # scene.init_pose2d_rgb() ## for visualization
    scene.load_fit_pose3d() ## load all the 3d poses in memory

    output_path = os.path.join(args.output_path, 'coco_track') 
    os.makedirs(output_path, exist_ok=True)

    time_stamps = list(range(1, scene.total_time_fit_pose3d  + 1))

    for t in tqdm(time_stamps):
        scene.update(time_stamp=t)
        cameras = scene.exo_camera_names_with_mode + scene.ego_camera_names_with_mode

        poses3d = scene.get_poses3d()

        for (camera_name, camera_mode) in cameras:
            scene.set_view(camera_name=camera_name, camera_mode=camera_mode)
            poses2d = scene.get_projected_poses3d()
            image_path = scene.view_camera.get_image_path(t)
            image_width = scene.view_camera.image_width
            image_height = scene.view_camera.image_height

            if camera_name.startswith('cam'):
                image_path = '/'.join(image_path.split('/')[-6:]) ## create the relative image path
                coco_add_to(sequence_name, parent_sequence, COCO_EXO_OUTPUT, COCO_EXO_DETECTIONS, poses2d, poses3d, image_path, image_width, image_height, camera_name, camera_mode, t)

            elif camera_name.startswith('aria'):
                image_path = '/'.join(image_path.split('/')[-7:]) ## create the relative image path
                
                if camera_mode == 'rgb':
                    depth = scene.view_camera.get_depth(time_stamp=t)
                    depth_path = scene.view_camera.get_depth_path(time_stamp=t)
                    coco_add_to(sequence_name, parent_sequence, COCO_EGO_RGB_OUTPUT, COCO_EGO_RGB_DETECTIONS, poses2d, poses3d, image_path, image_width, image_height, camera_name, camera_mode, t, \
                        depth, extrinsics=scene.view_camera.extrinsics, depth_path=depth_path)

                else:
                    coco_add_to(sequence_name, parent_sequence, COCO_EGO_SLAM_OUTPUT, COCO_EGO_SLAM_DETECTIONS, poses2d, poses3d, image_path, image_width, image_height, camera_name, camera_mode, t)

    exo_lines = []
    for annotation in COCO_EXO_OUTPUT['annotations']:
        frame_id = annotation['image_id'] + 1 ## <frame_id> # starts from 1 but COCO style starts from 0,
        instance_id = annotation['instance_id']
        x1 = annotation['bbox'][0]
        y1 = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]
        conf = 1
        class_id = 1
        visibility = 1
        line = '{} {} {} {} {} {} {} {} {}\n'.format(frame_id, instance_id, x1, y1, w, h, conf, class_id, visibility)
        exo_lines.append(line)

    ego_rgb_lines = []
    for annotation in COCO_EGO_RGB_OUTPUT['annotations']:
        frame_id = annotation['image_id'] + 1 ## <frame_id> # starts from 1 but COCO style starts from 0,
        instance_id = annotation['instance_id']
        x1 = annotation['bbox'][0]
        y1 = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]
        conf = 1
        class_id = 1
        visibility = 1
        line = '{} {} {} {} {} {} {} {} {}\n'.format(frame_id, instance_id, x1, y1, w, h, conf, class_id, visibility)
        ego_rgb_lines.append(line)

    ego_slam_lines = []
    for annotation in COCO_EGO_SLAM_OUTPUT['annotations']:
        frame_id = annotation['image_id'] + 1 ## <frame_id> # starts from 1 but COCO style starts from 0,
        instance_id = annotation['instance_id']
        x1 = annotation['bbox'][0]
        y1 = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]
        conf = 1
        class_id = 1
        visibility = 1
        line = '{} {} {} {} {} {} {} {} {}\n'.format(frame_id, instance_id, x1, y1, w, h, conf, class_id, visibility)
        ego_slam_lines.append(line)

    save_txt_path = os.path.join(output_path, 'gt_exo.txt')
    with open(save_txt_path, 'w+') as f:
        f.writelines(exo_lines)

    save_txt_path = os.path.join(output_path, 'gt_ego_rgb.txt')
    with open(save_txt_path, 'w+') as f:
        f.writelines(ego_rgb_lines)

    save_txt_path = os.path.join(output_path, 'gt_ego_slam.txt')
    with open(save_txt_path, 'w+') as f:
        f.writelines(ego_slam_lines)

    ##-------------------------------------------------------------
    save_keypoints_path = os.path.join(output_path, 'tracking_exo.json')
    with open(save_keypoints_path, 'w+') as output_json_file:
        json.dump(COCO_EXO_OUTPUT, output_json_file, indent=4, cls=NpEncoder)

    save_detections_path = os.path.join(output_path, 'detections_exo.pkl')
    with open(save_detections_path, 'wb') as f:
        pickle.dump(COCO_EXO_DETECTIONS, f)

    save_keypoints_path = os.path.join(output_path, 'tracking_ego_rgb.json')
    with open(save_keypoints_path, 'w+') as output_json_file:
        json.dump(COCO_EGO_RGB_OUTPUT, output_json_file, indent=4, cls=NpEncoder)

    save_detections_path = os.path.join(output_path, 'detections_ego_rgb.pkl')
    with open(save_detections_path, 'wb') as f:
        pickle.dump(COCO_EGO_RGB_DETECTIONS, f)

    save_keypoints_path = os.path.join(output_path, 'tracking_ego_slam.json')
    with open(save_keypoints_path, 'w+') as output_json_file:
        json.dump(COCO_EGO_SLAM_OUTPUT, output_json_file, indent=4, cls=NpEncoder)

    save_detections_path = os.path.join(output_path, 'detections_ego_slam.pkl')
    with open(save_detections_path, 'wb') as f:
        pickle.dump(COCO_EGO_SLAM_DETECTIONS, f)

    return


##------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()