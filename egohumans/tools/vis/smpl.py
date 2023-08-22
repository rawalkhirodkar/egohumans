import _init_paths
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from colour import Color
import cv2
import matplotlib.pyplot as plt
from cycler import cycle
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import mmcv

from datasets.aria_camera import AriaCamera
from datasets.aria_human import AriaHuman
from datasets.ego_exo_scene import EgoExoScene

from configs import cfg
from configs import update_config
from multiprocessing import Pool
from copy import deepcopy

##------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Visualization of extrinsics of camera parameters.')
    parser.add_argument('--sequence_path', action='store', help='the path to the sequence for visualization')
    parser.add_argument('--output_path', action='store', help='the path to the sequence for visualization')
    parser.add_argument('--start_time', default=1, help='start time')
    parser.add_argument('--end_time', default=-1, help='end time')

    args = parser.parse_args()
    sequence_path = args.sequence_path
    sequence_name = sequence_path.split('/')[-1]
    parent_sequence = sequence_name.split('_')[-1]
    config_file = os.path.join(_init_paths.root_path, 'configs', parent_sequence, '{}.yaml'.format(sequence_name))
    update_config(cfg, config_file)

    output_path = os.path.join(args.output_path, 'smpl') 
    vis_output_path = os.path.join(args.output_path, 'vis_smpl') 
    print('sequence at {}'.format(sequence_path))

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(vis_output_path, exist_ok=True)

    scene = EgoExoScene(cfg=cfg, root_dir=sequence_path)

    ##-----------------smpl-----------------------
    scene.load_smpl() ## load all the 3d poses in memory

    ##-----------------visualize-----------------------
    ## we start with t=1
    start_time = int(args.start_time)
    end_time = int(args.end_time)
    
    if end_time == -1:
        end_time = scene.total_time_smpl  

    time_stamps = list(range(start_time, end_time + 1))

    ##-----------------visualize-----------------------
    for t in tqdm(time_stamps):
        scene.update(time_stamp=t)
        cameras = scene.exo_camera_names_with_mode + scene.ego_camera_names_with_mode
        smpl = {human_name: scene.aria_humans[human_name].smpl for human_name in scene.aria_human_names}
        for (camera_name, camera_mode) in cameras:
            
            # ##-------------visualize the pose-------------------    
            # ## skip cameras if not in vis list
            # if scene.cfg.SMPL.VIS_CAMERAS != [] and camera_name not in scene.cfg.SMPL.VIS_CAMERAS:
            #     continue

            scene.set_view(camera_name=camera_name, camera_mode=camera_mode)
            save_dir = os.path.join(vis_output_path, scene.viewer_name, scene.view_camera_type)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, '{:05d}.jpg'.format(t))
            scene.draw_smpl(smpl, save_path)

    print('done, start:{}, end:{} -- both inclusive!'.format(start_time, end_time))

    return



##------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()