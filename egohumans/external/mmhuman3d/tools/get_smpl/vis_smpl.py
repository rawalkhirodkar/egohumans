import argparse
import os
import time

import mmcv
import numpy as np
import torch
from pathlib import Path
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.evaluation import keypoint_mpjpe
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mmhuman3d.data.data_structures import HumanData
from mmhuman3d.models.registrants.builder import build_registrant
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY

####---------------------------------------------------
MESH_COLORS = {
    'pink': np.array([197, 27, 125]),
    'light_pink': np.array([233, 163, 201]),
    'light_green': np.array([161, 215, 106]),
    'green': np.array([77, 146, 33]),
    'red': np.array([215, 48, 39]),
    'light_red': np.array([252, 146, 114]),
    'light_orange': np.array([252, 141, 89]),
    'purple': np.array([118, 42, 131]),
    'light_purple': np.array([175, 141, 195]),
    'light_blue': np.array([145, 191, 219]),
    'blue': np.array([69, 117, 180]),
    'gray': np.array([130, 130, 130]),
    'white': np.array([255, 255, 255]),
    'turkuaz': np.array([50, 134, 204]),
    'orange': np.array([205, 133, 51]),
    'light_yellow': np.array([255, 255, 224]),
}

####---------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d smplify tool')
    parser.add_argument(
        '--recording',
        help='input type')
    parser.add_argument(
        '--gender',
        choices=['neutral', 'male', 'female'],
        default='neutral',
        help='gender of SMPL model')
    parser.add_argument(
        '--time',
        default='2500',
        help='time stamp')
    parser.add_argument('--config', help='smplify config file path')
    parser.add_argument('--body_model_dir', help='body models file path')
    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument(
        '--show_path', help='directory to save rendered images or video')
    args = parser.parse_args()
    return args

####---------------------------------------------------
def vis_turntable(verts, body_model_config, show_path, palette, resolution=30):
    azimuths = np.arange(0, 360 + resolution, resolution)
    vis_dir = os.path.join(show_path, 'images')

    for azim in azimuths:
        visualize_smpl_pose(
            verts=verts,
            body_model_config=body_model_config,
            output_path=vis_dir,
            orbit_speed=1,
            elev=180,
            azim=azim,
            palette=palette,
            overwrite=True)

        cmd = 'mogrify -rotate -180 {}'.format(os.path.join(vis_dir, '000000.png')); os.system(cmd)
        cmd = 'mv {} {}'.format(os.path.join(vis_dir, '000000.png'), os.path.join(vis_dir, '{:06d}.png'.format(azim))); os.system(cmd)

    cmd = 'convert -delay 100 -loop 0 {}/*.png {}/output.gif'.format(vis_dir, show_path)
    os.system(cmd)

    return


####---------------------------------------------------
def vis_smpl(smplify_config, body_model_config, smpl_dir, timestamp, show_path):
    person_smpl_names = sorted([x for x in os.listdir(smpl_dir) if x.startswith(timestamp) and x.endswith('.npz')])
    person_smpl_datas = [HumanData.fromfile(os.path.join(smpl_dir, person_smpl_name)) for person_smpl_name in person_smpl_names]

    person_idxs = [int(x.replace('.npz', '').replace('{}_'.format(timestamp), '')) for x in person_smpl_names]
    person_colors = [MESH_COLORS['light_red'], MESH_COLORS['light_green'], MESH_COLORS['light_blue']]

    palette = np.concatenate([person_colors[person_idx].reshape(1, -1) for person_idx in person_idxs], axis=0)
    assert(len(person_smpl_datas) > 0)
    temp = [    person_smpl_data['smpl']['vertices'].unsqueeze(dim=1) + \
                person_smpl_data['meta']['root_3d'].astype('float32') \
                for person_smpl_data in person_smpl_datas \
            ]
    verts = torch.cat(temp, dim=1)
    vis_turntable(verts, body_model_config, show_path, palette, resolution=30)
    
    return



####---------------------------------------------------
def main():
    args = parse_args()
    smplify_config = mmcv.Config.fromfile(args.config)
    body_model_dir = os.path.dirname(args.body_model_dir.rstrip('/'))
    smpl_dir = os.path.join(args.recording, 'smpl')
    timestamp = args.time
    show_path = args.show_path

    # create body model
    body_model_config = dict(
        type=smplify_config.body_model.type.lower(),
        gender=args.gender,
        num_betas=args.num_betas,
        model_path=args.body_model_dir,
        batch_size=1,
    )

    body_model_config.update(
            model_path=body_model_dir,
            model_type=smplify_config.body_model.type.lower())

    vis_smpl(smplify_config, body_model_config, smpl_dir, timestamp, show_path)

    return

if __name__ == '__main__':
    main()
