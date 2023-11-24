import _init_paths
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import pickle
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
def coco_append_to(COCO_OUTPUT, COCO_DETECTIONS, coco_path, coco_det_path):
    with open(coco_path, 'r') as f:
        coco_data = json.loads(f.read())

    big_sequence_name = coco_path.split('/')[-4]
    sequence_name = coco_path.split('/')[-3]

    video_id_to_new_video_id = {}
    for video_info in coco_data['videos']:
        video_id = video_info['id'] ## old video id
        new_video_id = COCO_OUTPUT['video_num'] + 1 ## we start with 1
        COCO_OUTPUT['video_num'] += 1
        video_id_to_new_video_id[video_id] = new_video_id

        video_info['id'] = new_video_id
        COCO_OUTPUT['videos'].append(video_info)

    image_id_to_new_image_id = {}
    for image_info in coco_data['images']:
        image_id = image_info['id']
        new_image_id = COCO_OUTPUT['image_num']
        COCO_OUTPUT['image_num'] += 1
        image_id_to_new_image_id[image_id] = new_image_id

        image_info['video_id'] = video_id_to_new_video_id[image_info['video_id']]
        image_info['id'] = new_image_id

        COCO_OUTPUT['images'].append(image_info)

    for annotation_info in coco_data['annotations']:
        annotation_info['id'] = COCO_OUTPUT['instance_num']
        COCO_OUTPUT['instance_num'] += 1
        annotation_info['video_id'] = video_id_to_new_video_id[annotation_info['video_id']]
        annotation_info['image_id'] = image_id_to_new_image_id[annotation_info['image_id']]

        COCO_OUTPUT['annotations'].append(annotation_info)

    ##----------------------------------------
    with open(coco_det_path, 'rb') as f:
        coco_det_data = pickle.load(f)

    for image_path in coco_det_data['det_bboxes'].keys():
        COCO_DETECTIONS['det_bboxes'][image_path] = coco_det_data['det_bboxes'][image_path]

    for image_path in coco_det_data['keypoints_3d'].keys():
        COCO_DETECTIONS['keypoints_3d'][image_path] = coco_det_data['keypoints_3d'][image_path]

    for image_path in coco_det_data['roots_3d'].keys():
        COCO_DETECTIONS['roots_3d'][image_path] = coco_det_data['roots_3d'][image_path]

    return


##------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Visualization of extrinsics of camera parameters.')
    parser.add_argument('--big_sequences', action='store', help='the path to the sequence for visualization')
    parser.add_argument('--sequences', action='store', help='the path to the sequence for visualization')
    parser.add_argument('--root_dir', action='store', help='the path to the sequence for visualization')
    parser.add_argument('--output_path', action='store', help='the path to the sequence for visualization')

    args = parser.parse_args()

    big_sequences = args.big_sequences.split(':')
    sequences = args.sequences.split(':')
    root_dir = args.root_dir

    for big_sequence, sequence in zip(big_sequences, sequences):
        print('sequence:', big_sequence, sequence)

        ##---------------append exo data--------------------------
        exo_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco_track', 'tracking_exo.json')
        exo_det_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco_track', 'detections_exo.pkl')
        coco_append_to(COCO_EXO_OUTPUT, COCO_EXO_DETECTIONS, exo_path, exo_det_path)

        ##---------------append ego rgb data--------------------------
        ego_rgb_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco_track', 'tracking_ego_rgb.json')
        ego_rgb_det_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco_track', 'detections_ego_rgb.pkl')
        coco_append_to(COCO_EGO_RGB_OUTPUT, COCO_EGO_RGB_DETECTIONS, ego_rgb_path, ego_rgb_det_path)

        ##---------------append ego slam data--------------------------
        ego_slam_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco_track', 'tracking_ego_slam.json')
        ego_slam_det_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco_track', 'detections_ego_slam.pkl')
        coco_append_to(COCO_EGO_SLAM_OUTPUT, COCO_EGO_SLAM_DETECTIONS, ego_slam_path, ego_slam_det_path)

    ###-------------------save------------------------------
    output_path = os.path.join(args.output_path, 'coco_track') 
    os.makedirs(output_path, exist_ok=True)

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