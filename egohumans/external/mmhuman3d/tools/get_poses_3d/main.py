import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import cv2
import matplotlib.pyplot as plt
from cycler import cycle
from pathlib import Path
from aria_camera import AriaCamera, read_calibration
from triangulation import triangulation, read_points 
from mpl_toolkits.mplot3d import Axes3D
from utils import COCO_KP_CONNECTIONS, write_array_to_file, read_array_from_file, write_for_smpl_fitting

np.random.seed(0)

##------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Visualization of extrinsics of camera parameters.')
parser.add_argument('--recording_path', action='store', help='the path to the sequence for visualization')
parser.add_argument('--time', action='store', help='time')
parser.add_argument('--output_path', action='store', help='output path')


##------------------------------------------------------------------------------------
def triangulation_runner(time_string, calibration_file, images_path, correspondences_path, target_person_idx, multiview_list, vis=False):
    ##-------------load calibration--------------------
    calibration_file_id = (calibration_file.split('/')[-1]).replace('.txt', '')
    calibration_data = read_calibration(calibration_file)

    person_ids = list(calibration_data.keys())
    persons_data = {person_idx:{} for person_idx in range(len(person_ids))} ## 0, 1, 2

    for person_idx, person_id in enumerate(person_ids):
        aria_camera = calibration_data[person_id]

        ##----get image path----------
        rgb_image_path = os.path.join(images_path, '{}_{}_0.jpg'.format(calibration_file_id, person_idx))
        rgb_image = cv2.imread(rgb_image_path)

        left_image_path = os.path.join(images_path, '{}_{}_1.jpg'.format(calibration_file_id, person_idx))
        left_image = cv2.imread(left_image_path)

        right_image_path = os.path.join(images_path, '{}_{}_2.jpg'.format(calibration_file_id, person_idx))
        right_image = cv2.imread(right_image_path)

        ###-----------rgb cam extrinsics-------------
        rgb_intrinsics = aria_camera['rgb']['intrinsics']
        rgb_extrinsics = aria_camera['rgb']['extrinsics']
        rgb_extrinsics = np.concatenate([rgb_extrinsics, [[0, 0, 0, 1]]], axis=0) ## 4 x 4
        rgb_cam = AriaCamera(person_id=person_id, type='rgb', intrinsics=rgb_intrinsics, extrinsics=rgb_extrinsics)

        ###-----------left cam extrinsics-------------
        left_intrinsics = aria_camera['left']['intrinsics']
        left_extrinsics = aria_camera['left']['extrinsics']
        left_extrinsics = np.concatenate([left_extrinsics, [[0, 0, 0, 1]]], axis=0) ## 4 x 4
        left_cam = AriaCamera(person_id=person_id, type='left', intrinsics=left_intrinsics, extrinsics=left_extrinsics)
        
        ###-----------right cam extrinsics-------------
        right_intrinsics = aria_camera['right']['intrinsics']
        right_extrinsics = aria_camera['right']['extrinsics']
        right_extrinsics = np.concatenate([right_extrinsics, [[0, 0, 0, 1]]], axis=0) ## 4 x 4
        right_cam = AriaCamera(person_id=person_id, type='right', intrinsics=right_intrinsics, extrinsics=right_extrinsics)
        
        ###--------read the 2d points-----
        points_2d_rgb = read_points(os.path.join(correspondences_path, '{}_{}_0.txt'.format(calibration_file_id, person_idx)), target_person_idx)
        points_2d_left = read_points(os.path.join(correspondences_path, '{}_{}_1.txt'.format(calibration_file_id, person_idx)), target_person_idx)
        points_2d_right = read_points(os.path.join(correspondences_path, '{}_{}_2.txt'.format(calibration_file_id, person_idx)), target_person_idx)

        ###------ store data-----
        person_data = {'rgb_cam': rgb_cam, 'left_cam': left_cam, 'right_cam': right_cam, \
                        'points_2d_rgb': points_2d_rgb, 'points_2d_left': points_2d_left, 'points_2d_right': points_2d_right, \
                        'rgb_image': rgb_image, 'left_image': left_image, 'right_image': right_image, \
                        }
        persons_data[person_idx] = person_data

    ##----choice of views------
    # multiview_list = [(1, 'rgb'), (2, 'rgb')] ## person_idx, type
    multiview_evidence = []

    for selected_view in multiview_list:
        person_idx = selected_view[0]
        view_type = selected_view[1]

        image = persons_data[person_idx]['{}_image'.format(view_type)]
        points_2d = persons_data[person_idx]['points_2d_{}'.format(view_type)]
        cam = persons_data[person_idx]['{}_cam'.format(view_type)]

        singleview_evidence = {'cam': cam, 'points_2d': points_2d}
        multiview_evidence.append(singleview_evidence)

        ####----------vis---------------
        if vis == True:
            plt.imshow(image[:,:,[2,1,0]])
            plt.scatter(points_2d[:,0], points_2d[:,1])
            plt.show()

    ###---------trinagulate-------
    points_3d_prime = triangulation(multiview_evidence=multiview_evidence)

    return points_3d_prime

##------------------------------------------------------------------------------------
def main(time_string, calibration_file, images_path, correspondences_path, output_path, vis=False):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    ##--------person 1--------------
    target_person_idx = 1 ## green shirt
    multiview_list = [(0, 'rgb'), (0, 'left'), (2, 'rgb'), (2, 'left'), (2, 'right')] ## source
    points_3d = triangulation_runner(time_string, calibration_file, images_path, correspondences_path, \
                    target_person_idx=target_person_idx, multiview_list=multiview_list, vis=vis) ## green shirt
    save_path = os.path.join(output_path, '{}_{}.json'.format(time_string, target_person_idx))
    write_array_to_file(points_3d, save_path)
    save_path = os.path.join(output_path, '{}_{}.npz'.format(time_string, target_person_idx))
    write_for_smpl_fitting(points_3d, save_path)

    p1_points_3d = points_3d

    ##--------person 2--------------
    target_person_idx = 2 ## dark blue shirt
    multiview_list = [(0, 'rgb'), (0, 'left'), (1, 'rgb'), (1, 'left')] ## source
    points_3d = triangulation_runner(time_string, calibration_file, images_path, correspondences_path, \
                    target_person_idx=target_person_idx, multiview_list=multiview_list, vis=vis) ## dark blue shirt
    save_path = os.path.join(output_path, '{}_{}.json'.format(time_string, target_person_idx))
    write_array_to_file(points_3d, save_path)
    save_path = os.path.join(output_path, '{}_{}.npz'.format(time_string, target_person_idx))
    write_for_smpl_fitting(points_3d, save_path)

    p2_points_3d = points_3d

    ##---------------visualize---------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    connections = COCO_KP_CONNECTIONS

    for _c in connections:
        p3ds = p1_points_3d
        ax.plot(xs=[p3ds[_c[0],0], p3ds[_c[1],0]], ys=[p3ds[_c[0],1], p3ds[_c[1],1]], zs=[p3ds[_c[0],2], p3ds[_c[1],2]], c='green')

        p3ds = p2_points_3d
        ax.plot(xs=[p3ds[_c[0],0], p3ds[_c[1],0]], ys=[p3ds[_c[0],1], p3ds[_c[1],1]], zs=[p3ds[_c[0],2], p3ds[_c[1],2]], c='blue')


    plt.show()


    return


##------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    recording_path = args.recording_path
    output_path = args.output_path
    time = int(args.time)

    time_string = '{0:05d}'.format(time)

    print('recording at {}'.format(recording_path))
    
    calibration_path = os.path.join(recording_path, 'calib')
    images_path = os.path.join(recording_path, 'images')
    correspondences_path = os.path.join(recording_path, 'poses_2d') 
    output_path = os.path.join(output_path, 'poses_3d') ## save the 3D keypoints

    calibration_file = os.path.join(calibration_path, '{}.txt'.format(time_string))
    main(time_string=time_string, calibration_file=calibration_file, \
        images_path=images_path, correspondences_path=correspondences_path, \
        output_path=output_path, vis=False)
    print('done')
















