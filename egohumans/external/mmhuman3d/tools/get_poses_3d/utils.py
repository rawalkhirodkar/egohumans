import numpy as np
import torch
import json
from json import JSONEncoder



###--------------------------------------------------------------
# https://github.com/open-mmlab/mmhuman3d/blob/main/tools/smplify.py
def write_for_smpl_fitting(poses_3d, save_path):
    left_hip_index = COCO_KP_ORDER.index('left_hip')
    right_hip_index = COCO_KP_ORDER.index('right_hip')

    root_3d = (poses_3d[left_hip_index, :] + poses_3d[right_hip_index, :])/2
    centered_poses_3d = poses_3d - root_3d ## K x 3

    # flip the poses upside down
    # centered_poses_3d[:, 1] *= -1 

    ## turn the poses to face the camera
    # centered_poses_3d[:,0] *= -1 

    human_data_format = {   \
                        'keypoints3d': np.expand_dims(centered_poses_3d, axis=0), \
                        'keypoints3d_mask': np.ones(centered_poses_3d.shape[0]), \
                        'meta':{'root_3d': np.expand_dims(root_3d, axis=0)} ## time x 3
                        } # time x num_keypoints x 3, mask of len num_keypoints
    
    np.savez_compressed(save_path, **human_data_format) 

    return

###--------------------------------------------------------------
class ArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

## to save 3d poses
def write_array_to_file(poses_3d, file):
    data = {'poses_3d': poses_3d}
    with open(file, 'w') as f:
        json.dump(data, f, cls=ArrayEncoder, indent=4, sort_keys=True)

    return

def read_array_from_file(file):
    with open(file, 'r') as f:
        data = json.load(f)
    poses_3d = np.asarray(data['poses_3d'])
    return poses_3d

###--------------------------------------------------------------
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

###--------------------------------------------------------------
