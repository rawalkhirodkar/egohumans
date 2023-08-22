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
        '--input',
        help=('input file path.'
              'Input shape should be [N, J, D] or [N, M, J, D],'
              ' where N is the sequence length, M is the number of persons,'
              ' J is the number of joints and D is the dimension.'))
    parser.add_argument(
        '--input_type',
        choices=['keypoints2d', 'keypoints3d'],
        default='keypoints3d',
        help='input type')
    parser.add_argument(
        '--J_regressor',
        type=str,
        default=None,
        help='the path of the J_regressor')
    parser.add_argument(
        '--keypoint_type',
        default='human_data',
        help='the source type of input keypoints')
    parser.add_argument('--config', help='smplify config file path')
    parser.add_argument('--body_model_dir', help='body models file path')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_betas', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument(
        '--use_one_betas_per_video',
        action='store_true',
        help='use one betas to keep shape consistent through a video')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for smplify')
    parser.add_argument(
        '--gender',
        choices=['neutral', 'male', 'female'],
        default='neutral',
        help='gender of SMPL model')
    parser.add_argument('--output', help='output result file')
    parser.add_argument(
        '--show_path', help='directory to save rendered images or video')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite if there is already a result file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    smplify_config = mmcv.Config.fromfile(args.config)
    assert smplify_config.body_model.type.lower() in ['smpl', 'smplx']
    assert smplify_config.type.lower() in ['smplify', 'smplifyx']

    # set cudnn_benchmark
    if smplify_config.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    gt_human_data = HumanData.fromfile(args.input)
    keypoints_src = gt_human_data[args.input_type]
    keypoints_src_mask = gt_human_data[args.input_type + '_mask']

    ###---------------add more weights to stuff-----------------
    src_keypoints_order = KEYPOINTS_FACTORY[args.keypoint_type]
    dst_keypoints_order = KEYPOINTS_FACTORY[smplify_config.body_model['keypoint_dst']]

    keypoints_src_mask[src_keypoints_order.index('left_wrist')] = 2
    keypoints_src_mask[src_keypoints_order.index('right_wrist')] = 2

    keypoints_src_mask[src_keypoints_order.index('left_elbow')] = 2
    keypoints_src_mask[src_keypoints_order.index('right_elbow')] = 2

    ###-------------------------------------------------------
    if args.input_type == 'keypoints2d':
        assert keypoints_src.shape[-1] in {2, 3}
    elif args.input_type == 'keypoints3d':
        assert keypoints_src.shape[-1] in {3, 4}
        keypoints_src = keypoints_src[..., :3]
    else:
        raise KeyError('Only support keypoints2d and keypoints3d')

    keypoints, mask = convert_kps(
        keypoints_src,
        mask=keypoints_src_mask,
        src=args.keypoint_type,
        dst=smplify_config.body_model['keypoint_dst'])
    keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

    ###------vis kp3d----------
    vis_keypoints_src = keypoints_src.copy()
    vis_keypoints_src[:, :, 1] *= -1 ## flip the y axis for better vis
    visualize_kp3d(
            kp3d=vis_keypoints_src,
            output_path=args.show_path,
        )

    batch_size = args.batch_size if args.batch_size else keypoints.shape[0]

    keypoints = torch.tensor(keypoints, dtype=torch.float32, device=device)
    keypoints_conf = torch.tensor(
        keypoints_conf, dtype=torch.float32, device=device)

    if args.input_type == 'keypoints3d':
        human_data = dict(
            keypoints3d=keypoints, keypoints3d_conf=keypoints_conf)
    elif args.input_type == 'keypoints2d':
        human_data = dict(
            keypoints2d=keypoints, keypoints2d_conf=keypoints_conf)
    else:
        raise TypeError(f'Unsupported input type: {args.input_type}')

    # create body model
    body_model_config = dict(
        type=smplify_config.body_model.type.lower(),
        gender=args.gender,
        num_betas=args.num_betas,
        model_path=args.body_model_dir,
        batch_size=batch_size,
    )

    if args.J_regressor is not None:
        body_model_config.update(dict(joints_regressor=args.J_regressor))

    if smplify_config.body_model.type.lower() == 'smplx':
        body_model_config.update(
            dict(
                use_face_contour=True,  # 127 -> 144
                use_pca=False,  # current vis do not supports use_pca
            ))

    smplify_config.update(
        dict(
            body_model=body_model_config,
            use_one_betas_per_video=args.use_one_betas_per_video,
            num_epochs=args.num_epochs))

    smplify = build_registrant(dict(smplify_config))

    # run SMPLify(X)
    t0 = time.time()
    smplify_output, smplify_output_per_epoch = smplify(**human_data, return_joints=True, return_verts=True)
    t1 = time.time()
    print(f'Time:  {t1 - t0:.2f} s')

    # test MPJPE
    pred = smplify_output['joints'].cpu().numpy()
    gt = keypoints.cpu().numpy()
    mask = mask.reshape(1, -1).repeat(gt.shape[0], axis=0).astype(bool)
    mpjpe = keypoint_mpjpe(pred=pred, gt=gt, mask=mask)

    # get smpl parameters directly from smplify output
    poses = {k: v.detach().cpu() for k, v in smplify_output.items()}
    smplify_results = HumanData(dict(smpl=poses))

    ##-------------add gt info-----------------
    smplify_results['meta'] = { \
                    'keypoints3d': gt_human_data['keypoints3d'],    \
                    'keypoints3d_mask': gt_human_data['keypoints3d_mask'],  \
                    'root_3d': gt_human_data['meta']['root_3d'].astype('float64'),    \
                    'src': args.input,  \
                    }

    ##-----------------------------------
    if args.output is not None:
        print(f'Dump results to {args.output}')
        output_dir = '/'.join(args.output.split('/')[:-1])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        smplify_results.dump(args.output, overwrite=args.overwrite)

    ##-----write per epoch results--------
    for epoch in smplify_output_per_epoch.keys():
        poses_per_epoch = {k: v.detach().cpu() for k, v in smplify_output_per_epoch[epoch].items()}

        visualize_smpl_pose(
            poses=poses_per_epoch,
            body_model_config=body_model_config,
            output_path=os.path.join(args.show_path, 'epochs', str(epoch)),
            orbit_speed=1.0,
            plot_kps=True,
            kp3d=keypoints.cpu().numpy(),
            elev=180,
            azim=0,
            palette=MESH_COLORS['light_blue'],
            overwrite=True)

    ##-----------vis------------------
    if args.show_path is not None:
        # visualize smpl pose
        body_model_dir = os.path.dirname(args.body_model_dir.rstrip('/'))
        body_model_config.update(
            model_path=body_model_dir,
            model_type=smplify_config.body_model.type.lower())
        import pdb; pdb.set_trace()
        visualize_smpl_pose(
            poses=poses,
            body_model_config=body_model_config,
            output_path=args.show_path,
            orbit_speed=1,
            elev=180,
            azim=0,
            palette=MESH_COLORS['light_blue'],
            overwrite=True)

    ##-----------move images and rotate------------
    for epoch in smplify_output_per_epoch.keys():
        source_path = os.path.join(args.show_path, 'epochs', str(epoch), '000000.png')
        target_path = os.path.join(args.show_path, 'epochs', '{:04d}.png'.format(epoch + 1))
        os.system("mv {} {}".format(source_path, target_path))
        os.system("rm -rf {}".format(os.path.join(args.show_path, 'epochs', str(epoch))))
    
    ## rotate    
    cmd = 'mogrify -rotate -180 {}/*.png'.format(os.path.join(args.show_path, 'epochs'))
    os.system(cmd)
    cmd = 'mogrify -rotate -180 {}'.format(os.path.join(args.show_path, '000000.png'))
    os.system(cmd)

    ## make a gif
    cmd = 'convert -delay 100 -loop 0 {}/*.png {}/output.gif'.format(\
                        os.path.join(args.show_path, 'epochs'), \
                        args.show_path)
    
    os.system(cmd)

    print(f'SMPLify MPJPE: {mpjpe * 1000:.2f} mm')
    return

if __name__ == '__main__':
    main()
