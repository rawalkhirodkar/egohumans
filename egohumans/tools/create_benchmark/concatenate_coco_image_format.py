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

#------------------------------------------------------------------------------------
INFO = {
    "description": "EgoHumans",
    "url": "https://github.com/rawalkhirodkar/egohumans",
    "version": "1.0",
    "year": 2023,
    "contributor": "rawalkhirodkar",
    "date_created": datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

OTHER_CATEGORIES = [{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]
CATEGORIES = [
    {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"
        ],
        "skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]
    }
    
]
CATEGORIES += OTHER_CATEGORIES

COCO_KP_ORDER = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines

COCO_KP_CONNECTIONS = kp_connections(COCO_KP_ORDER)

COCO_EXO_OUTPUT = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
        "image_num": 0,
        "instance_num": 0,
    }

COCO_EGO_RGB_OUTPUT = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
        "image_num": 0,
        "instance_num": 0,
    }


COCO_EGO_SLAM_OUTPUT = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
        "image_num": 0,
        "instance_num": 0,
    }

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
def coco_append_to(COCO_OUTPUT, coco_path):
    with open(coco_path, 'r') as f:
        coco_data = json.loads(f.read())

    image_id_to_new_image_id = {}

    for image_info in coco_data['images']:
        image_id = image_info['id']
        new_image_id = COCO_OUTPUT['image_num']
        COCO_OUTPUT['image_num'] += 1
        image_id_to_new_image_id[image_id] = new_image_id

        image_info['id'] = new_image_id
        COCO_OUTPUT['images'].append(image_info)

    # import pdb; pdb.set_trace()
    for annotation_info in coco_data['annotations']:
        annotation_info['id'] = COCO_OUTPUT['instance_num']
        COCO_OUTPUT['instance_num'] += 1
        annotation_info['image_id'] = image_id_to_new_image_id[annotation_info['image_id']]
        COCO_OUTPUT['annotations'].append(annotation_info)

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
        exo_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco', 'person_keypoints_exo.json')
        coco_append_to(COCO_EXO_OUTPUT, exo_path)

        ##---------------append ego rgb data--------------------------
        exo_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco', 'person_keypoints_ego_rgb.json')
        coco_append_to(COCO_EGO_RGB_OUTPUT, exo_path)

        ##---------------append ego slam data--------------------------
        exo_path = os.path.join(root_dir, 'benchmark', big_sequence, sequence, 'coco', 'person_keypoints_ego_slam.json')
        coco_append_to(COCO_EGO_SLAM_OUTPUT, exo_path)

    ###-------------------save------------------------------
    output_path = os.path.join(args.output_path, 'coco') 
    os.makedirs(output_path, exist_ok=True)

    save_keypoints_path = os.path.join(output_path, 'person_keypoints_exo.json')
    with open(save_keypoints_path, 'w+') as output_json_file:
        json.dump(COCO_EXO_OUTPUT, output_json_file, indent=4, cls=NpEncoder)

    save_keypoints_path = os.path.join(output_path, 'person_keypoints_ego_rgb.json')
    with open(save_keypoints_path, 'w+') as output_json_file:
        json.dump(COCO_EGO_RGB_OUTPUT, output_json_file, indent=4, cls=NpEncoder)

    save_keypoints_path = os.path.join(output_path, 'person_keypoints_ego_slam.json')
    with open(save_keypoints_path, 'w+') as output_json_file:
        json.dump(COCO_EGO_SLAM_OUTPUT, output_json_file, indent=4, cls=NpEncoder)

    return


##------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()