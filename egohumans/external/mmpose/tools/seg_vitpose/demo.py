import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
from mmpose.apis import (inference_top_down_seg_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument('--mask-path', type=str, default='', help='Path to the binary mask')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    image_name = args.img

    # Create a binary mask using the mouse
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return

    # Calculate the bounding box from the mask
    y, x = np.where(mask == 255)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    bbox = [x_min, y_min, x_max, y_max, 1.0]

    # Create a person_results object with bbox and segmentation mask
    person_results = [{
        'bbox': bbox,
        'segmentation': mask
    }]

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    pose_results, returned_outputs = inference_top_down_seg_pose_model(
        pose_model,
        image_name,
        person_results,
        bbox_thr=0.0,  # Set to 0.0 since we are using a user-provided mask
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    if args.out_img_root == '':
        out_file = None
    else:
        os.makedirs(args.out_img_root, exist_ok=True)
        out_file = os.path.join(args.out_img_root, f'vis_{os.path.basename(args.img)}')

    # show the results
    vis_pose_result(
        pose_model,
        image_name,
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr,
        radius=args.radius,
        thickness=args.thickness,
        show=args.show,
        out_file=out_file)

if __name__ == '__main__':
    main()

