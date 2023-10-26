import numpy as np
import os
import cv2
import trimesh
import pickle
import json
import pycolmap
import collections
import matplotlib.pyplot as plt
from models.pose_estimator import PoseModel
from models.human_detector import DetectorModel

from .aria_human import AriaHuman
from .exo_camera import ExoCamera

from utils.triangulation import Triangulator
from utils.keypoints_info import COCO_KP_CONNECTIONS
from utils.keypoints_info import COCO_KP_ORDER

from models.fit_pose3d import compute_limb_length

from utils.icp import icp
from utils.transforms import linear_transform, fast_circle, slow_circle, plane_unit_normal
import pathlib
from pyntcloud import PyntCloud
import pandas as pd
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from pyntcloud.geometry.models.plane import Plane

try:
    ##---import mmhuman functions
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    from mmhuman3d.core.conventions.keypoints_mapping.coco import COCO_KEYPOINTS
    from mmhuman3d.utils.transforms import rotmat_to_aa

    from models.smpl_estimator import SMPLModel
    from models.smplify import SMPLify

except:
    print('Cannot import mmhuman3d!')

## needed for consistent pyntcloud plane fitting
np.random.seed(0) ## seed the numpy so that the results are reproducible
##------------------------------------------------------------------------------------
class EgoExoScene:
    def __init__(self, cfg, root_dir):
        start_time = cv2.getTickCount()
        self.cfg = cfg
        print(self.cfg)
        print('sequence: {}'.format(self.cfg.SEQUENCE))
        self.root_dir = root_dir
        self.exo_dir = os.path.join(self.root_dir, 'exo')
        self.ego_dir = os.path.join(self.root_dir, 'ego')    
        self.colmap_dir = os.path.join(self.root_dir, 'colmap', 'workplace')

        self.bbox_dir = os.path.join(self.root_dir, 'processed_data', 'bboxes')
        self.pose2d_dir = os.path.join(self.root_dir, 'processed_data', 'poses2d')
        self.segpose2d_dir = os.path.join(self.root_dir, 'processed_data', 'segposes2d')

        self.pose3d_dir = os.path.join(self.root_dir, 'processed_data', 'poses3d')
        self.refine_pose3d_dir = os.path.join(self.root_dir, 'processed_data', 'refine_poses3d')
        self.fit_pose3d_dir = os.path.join(self.root_dir, 'processed_data', 'fit_poses3d')
        self.init_smpl_dir = os.path.join(self.root_dir, 'processed_data', 'init_smpl')
        self.smpl_dir = os.path.join(self.root_dir, 'processed_data', 'smpl')
        self.smpl_collision_dir = os.path.join(self.root_dir, 'processed_data', 'smpl_collision')
        self.offshelf_segmentation_dir = os.path.join(self.root_dir, 'processed_data', 'offshelf_segmentation')
        self.segmentation_dir = os.path.join(self.root_dir, 'processed_data', 'segmentation')
        self.contact_pose3d_dir = os.path.join(self.root_dir, 'processed_data', 'contact_poses3d')

        if self.cfg.BBOX.SAVE_OFFSHELF_BOX_TO_DISK is True:
            self.offshelf_bbox_dir = os.path.join(self.root_dir, 'processed_data', 'offshelf_bboxes')
            pathlib.Path(self.offshelf_bbox_dir).mkdir(parents=True, exist_ok=True)

        ##---------------colmap things---------------------------
        ###-----------load the coordinate transofmrs---------------
        colmap_transforms_file = os.path.join(self.colmap_dir, 'colmap_from_aria_transforms.pkl') 
        inv_colmap_transforms_file = os.path.join(self.colmap_dir, 'aria_from_colmap_transforms.pkl') ## colmap to aria

        with open(colmap_transforms_file, 'rb') as handle:
            self.colmap_from_aria_transforms = pickle.load(handle) ## aria coordinate system to colmap

        with open(inv_colmap_transforms_file, 'rb') as handle:
            self.inv_colmap_from_aria_transforms = pickle.load(handle) ## colmap coordinate system to aria

        ##------transform from aria1 coordinate system to colmap
        self.anchor_ego_camera = self.cfg.CALIBRATION.ANCHOR_EGO_CAMERA
        self.primary_transform = self.colmap_from_aria_transforms[self.anchor_ego_camera]

        ##----------------load the scene point cloud-----------------
        ## measure the time for this function
        self.scene_vertices, self.scene_ground_vertices, self.ground_plane = self.load_scene_geometry()

        ##------------------------ego--------------------------
        self.aria_human_names = [human_name for human_name in sorted(os.listdir(self.ego_dir)) if human_name not in self.cfg.INVALID_ARIAS and human_name.startswith('aria')]

        self.aria_humans = {}
        for person_idx, aria_human_name in enumerate(self.aria_human_names):
            coordinate_transform = np.dot(
                                np.linalg.inv(self.colmap_from_aria_transforms[aria_human_name]), 
                                self.primary_transform
                            ) 

            self.aria_humans[aria_human_name] = AriaHuman(
                            cfg=cfg,
                            root_dir=self.ego_dir, human_name=aria_human_name, \
                            human_id=person_idx, ground_plane=self.ground_plane, \
                            coordinate_transform=coordinate_transform)

        self.total_time = self.aria_humans[self.aria_human_names[0]].total_time
        self.time_stamp = 0 ## 0 is an invalid time stamp, we start with 1

        ##------------------------exo--------------------------
        self.exo_camera_mapping = self.get_colmap_camera_mapping()
        self.exo_camera_names = [exo_camera_name for exo_camera_name in sorted(os.listdir(self.exo_dir)) if exo_camera_name not in self.cfg.INVALID_EXOS and exo_camera_name.startswith('cam')]
        self.colmap_reconstruction = pycolmap.Reconstruction(self.colmap_dir) ## this is the bottleneck
        self.exo_cameras = {exo_camera_name: ExoCamera(cfg=cfg, root_dir=self.exo_dir, colmap_dir=self.colmap_dir, \
                            exo_camera_name=exo_camera_name, coordinate_transform=self.primary_transform, reconstruction=self.colmap_reconstruction, \
                            exo_camera_mapping=self.exo_camera_mapping) \
                            for exo_camera_name in sorted(self.exo_camera_names)}  

        #-----check for total time-----
        for aria_human_name in self.aria_human_names:
            assert(self.aria_humans[aria_human_name].total_time == self.total_time)

        ##------------------------common---------------------
        self.view_camera = None

        ##-------------------used for triangulation-----------------
        self.ego_camera_names_with_mode = [(aria_human_name, camera_mode) \
                        for aria_human_name in self.aria_human_names \
                        for camera_mode in ['rgb', 'left', 'right']]

        self.exo_camera_names_with_mode = [(camera_name, camera_mode) \
                        for camera_name in self.exo_camera_names \
                        for camera_mode in ['rgb']]

        self.camera_names = self.ego_camera_names_with_mode + self.exo_camera_names_with_mode ##[(camera_name, camera_mode)...]
        self.cameras = {} ## all cameras

        for (camera_name, camera_mode) in self.camera_names:
            camera, view_type = self.get_camera(camera_name, camera_mode)
            self.cameras[(camera_name, camera_mode)] = camera

        ##-----------------used for smpl fitting---------------
        self.load_pose2d_flag = False
        self.load_pose3d_flag = False
        self.load_refine_pose3d_flag = False
        self.load_fit_pose3d_flag = False
        self.load_smpl_flag = False
        self.load_init_smpl_flag = False
        self.load_segmentation_flag = False
        self.load_smpl_collision_flag = False


        ##--------------------------------------------------------
        end_time = cv2.getTickCount()
        print('Scene load time: {:.4f} seconds'.format((end_time - start_time)/cv2.getTickFrequency()))
        return  

    ##--------------------------------------------------------
    def load_scene_geometry(self, max_dist=0.1):
        Point3D = collections.namedtuple(
                "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

        path = os.path.join(self.colmap_dir, 'points3D.txt')

        # https://github.com/colmap/colmap/blob/5879f41fb89d9ac71d977ae6cf898350c77cd59f/scripts/python/read_write_model.py#L308
        points3D = []
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    point3D_id = int(elems[0])
                    xyz = np.array(tuple(map(float, elems[1:4])))
                    rgb = np.array(tuple(map(int, elems[4:7])))
                    error = float(elems[7])
                    image_ids = np.array(tuple(map(int, elems[8::2])))
                    point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                    points3D.append(xyz.reshape(1, -1))

        points3D = np.concatenate(points3D, axis=0)
        points3D = linear_transform(points3D, np.linalg.inv(self.primary_transform)) ## convert to aria01 from colmap

        cloud = PyntCloud(pd.DataFrame(
                        # same arguments that you are passing to visualize_pcl
                        data=points3D,
                        columns=["x", "y", "z"]))

        is_floor = cloud.add_scalar_field("plane_fit", max_dist=max_dist)

        ground_points3D = points3D[cloud.points['is_plane'] == 1]

        ground_cloud = PyntCloud(pd.DataFrame(
                        # same arguments that you are passing to visualize_pcl
                        data=ground_points3D,
                        columns=["x", "y", "z"]))

        ground_plane = Plane()
        ground_plane.from_point_cloud(ground_cloud.xyz)
        ground_plane.get_equation()

        ###----------------------------------------
        if self.cfg.GEOMETRY.MANUAL_GROUND_PLANE_POINTS != "":
            array_string = self.cfg.GEOMETRY.MANUAL_GROUND_PLANE_POINTS.replace('array(', '').replace(')', '')
            ground_points3D = np.array(json.loads(array_string))

            ## enlarge the plane 10 times.
            ground_cloud = PyntCloud(pd.DataFrame(
                        # same arguments that you are passing to visualize_pcl
                        data=ground_points3D,
                        columns=["x", "y", "z"]))

            ground_plane = Plane()
            ground_plane.from_point_cloud(ground_cloud.xyz)
            a, b, c, d = ground_plane.get_equation() ## a, b, c, d: ax + by + cz + d = 0
            centroid = ground_points3D.mean(axis=0)
            centroid_up = centroid + 0.005*np.array([a, b, c])
            centroid_down = centroid - 0.005*np.array([a, b, c])

            ## equation of plane passing through origin and is parallel to our plane
            num_points = 1000
            bound_val = 100
            x = np.random.rand(num_points) * bound_val - bound_val/2 
            y = np.random.rand(num_points) * bound_val - bound_val/2
            z = -1*(a*x + b*y)/c

            ## bigger plane
            plane_points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1) + centroid
            plane_up_points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1) + centroid_up
            plane_down_points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1) + centroid_down

            # ground_points3D = np.concatenate([bigger_ground_points3D, ground_points3D], axis=0)
            ground_points3D = np.concatenate([plane_points, plane_up_points, plane_down_points], axis=0)

            ## add random noise
            ground_points3D += 0.1*np.random.rand(ground_points3D.shape[0], ground_points3D.shape[1])

        return points3D, ground_points3D, ground_plane

    ##--------------------------------------------------------
    def init_smpl(self):
        self.load_fit_pose3d_flag = True
        self.total_time_fit_pose3d = len([file for file in os.listdir(self.fit_pose3d_dir) if file.endswith('npy')])
        self.smpl_model = SMPLModel(cfg=self.cfg)
        self.smplify = SMPLify(cfg=self.cfg)
        return

    def init_smpl_trajectory(self):
        self.smpl_model = SMPLModel(cfg=self.cfg)
        self.smplify = SMPLify(cfg=self.cfg)

        ##----temporal-----
        self.total_time_fit_pose3d = len([file for file in os.listdir(self.fit_pose3d_dir) if file.endswith('npy')])
        time_stamps = list(range(1, self.total_time_fit_pose3d + 1))
        poses3d_trajectory = {aria_human_name:[] for aria_human_name in self.aria_human_names}

        for time_stamp in time_stamps:
            pose3d_path = os.path.join(self.refine_pose3d_dir, '{:05d}.npy'.format(time_stamp))
            poses3d = (np.load(pose3d_path, allow_pickle=True)).item()

            for aria_human_name in poses3d.keys():
                if aria_human_name in self.aria_human_names:
                    poses3d_trajectory[aria_human_name].append(poses3d[aria_human_name].reshape(1, -1, 4)) ## 1 x 17 x 4

        self.set_poses3d_trajectory(poses3d_trajectory) ## the pose trajectory for all the humans            
        return
    
    ## initialization function for smpl fitting without collision
    def init_smpl_trajectory_collision(self):
        self.smpl_model = SMPLModel(cfg=self.cfg)
        self.smplify = SMPLify(cfg=self.cfg)

        ##----temporal-----
        self.total_time_fit_pose3d = len([file for file in os.listdir(self.fit_pose3d_dir) if file.endswith('npy')])
        time_stamps = list(range(1, self.total_time_fit_pose3d + 1))
        poses3d_trajectory = {aria_human_name:[] for aria_human_name in self.aria_human_names}

        for time_stamp in time_stamps:
            pose3d_path = os.path.join(self.refine_pose3d_dir, '{:05d}.npy'.format(time_stamp))
            poses3d = (np.load(pose3d_path, allow_pickle=True)).item()

            for aria_human_name in poses3d.keys():
                if aria_human_name in self.aria_human_names:
                    poses3d_trajectory[aria_human_name].append(poses3d[aria_human_name].reshape(1, -1, 4)) ## 1 x 17 x 4

        self.set_poses3d_trajectory(poses3d_trajectory) ## the pose trajectory for all the humans  

        from models.smpl_estimator import get_smpl_faces
        self.smpl_faces = get_smpl_faces()
        self.total_time_smpl = len([file for file in os.listdir(self.smpl_dir) if file.endswith('npy')])   

        smpls_trajectory = {aria_human_name:[] for aria_human_name in self.aria_human_names}       
        
        for time_stamp in time_stamps:
            smpl_path = os.path.join(self.smpl_dir, '{:05d}.npy'.format(time_stamp))
            smpls = np.load(smpl_path, allow_pickle=True).item()

            for aria_human_name in smpls.keys():
                if aria_human_name in self.aria_human_names:
                    smpls_trajectory[aria_human_name].append(smpls[aria_human_name]) ## dict_keys(['global_orient', 'transl', 'body_pose', 'betas', 'epoch_loss', 'vertices', 'joints'])
        
        ## set the smpl trajectories for the humans
        for aria_human_name in smpls_trajectory.keys():
            self.aria_humans[aria_human_name].smpls_trajectory = smpls_trajectory[aria_human_name]
        
        return

    def load_smpl(self):
        self.load_smpl_flag = True
        self.total_time_smpl = len([file for file in os.listdir(self.smpl_dir) if file.endswith('npy')])
        
        from models.smpl_estimator import get_smpl_faces
        self.smpl_faces = get_smpl_faces()

        return

    def load_smpl_collision(self):
        self.load_smpl_collision_flag = True
        self.total_time_smpl_collision = len([file for file in os.listdir(self.smpl_collision_dir) if file.endswith('npy')])
        
        from models.smpl_estimator import get_smpl_faces
        self.smpl_faces = get_smpl_faces()

        return
    
    def load_init_smpl(self):
        self.load_init_smpl_flag = True
        self.total_time_smpl = len([file for file in os.listdir(self.init_smpl_dir) if file.endswith('npy')])
        
        from models.smpl_estimator import get_smpl_faces
        self.smpl_faces = get_smpl_faces()

        return

    ##-------------------------------------------------------
    def init_refine_smpl(self):
        self.load_smpl_flag = True
        self.total_time_smpl = len([file for file in os.listdir(self.smpl_dir) if file.endswith('npy')])
        return

    ##-------------------------------------------------------
    def init_refine_pose3d(self, override_contact_segpose3d=False):

        if self.cfg.POSE3D.USE_SEGPOSE2D == True and override_contact_segpose3d == False:
            self.pose3d_dir = self.contact_pose3d_dir

        self.total_time_pose3d = len([file for file in os.listdir(self.pose3d_dir) if file.endswith('npy')])

        ## load all the 3d poses in memory
        time_stamps = list(range(1, self.total_time_pose3d + 1))

        poses3d_trajectory = {aria_human_name:[] for aria_human_name in self.aria_human_names}

        missing_time_stamps = []

        for time_stamp in time_stamps:
            pose3d_path = os.path.join(self.pose3d_dir, '{:05d}.npy'.format(time_stamp))
            poses3d = (np.load(pose3d_path, allow_pickle=True)).item()

            if len(poses3d.keys()) != len(self.aria_human_names):
                missing_time_stamps.append(time_stamp)

            for aria_human_name in poses3d.keys():
                if aria_human_name not in self.aria_human_names:
                    print('skipping {}'.format(aria_human_name))
                    continue
                poses3d_trajectory[aria_human_name].append(poses3d[aria_human_name].reshape(1, -1, 4)) ## 1 x 17 x 4

        print('missing time stamps: {}'.format(missing_time_stamps))
        
        self.set_poses3d_trajectory(poses3d_trajectory)            

        return

    def load_refine_pose3d(self):
        self.load_refine_pose3d_flag = True
        self.total_time_refine_pose3d = len([file for file in os.listdir(self.refine_pose3d_dir) if file.endswith('npy')])
        return

    ##-------------------------------------------------------
    def init_fit_pose3d(self):
        self.total_time_refine_pose3d = len([file for file in os.listdir(self.refine_pose3d_dir) if file.endswith('npy')])

        ## load all the 3d poses in memory
        time_stamps = list(range(1, self.total_time_refine_pose3d + 1))

        poses3d_trajectory = {aria_human_name:[] for aria_human_name in self.aria_human_names}

        for time_stamp in time_stamps:
            pose3d_path = os.path.join(self.refine_pose3d_dir, '{:05d}.npy'.format(time_stamp))
            poses3d = (np.load(pose3d_path, allow_pickle=True)).item()

            for aria_human_name in poses3d.keys():
                if aria_human_name not in self.aria_human_names:
                    continue
                poses3d_trajectory[aria_human_name].append(poses3d[aria_human_name].reshape(1, -1, 4)) ## 1 x 17 x 4

        ##-----------------------------------
        self.set_poses3d_trajectory(poses3d_trajectory)            

        return

    def load_fit_pose3d(self):
        self.load_fit_pose3d_flag = True
        self.total_time_fit_pose3d = len([file for file in os.listdir(self.fit_pose3d_dir) if file.endswith('npy')])
        return

    ##-------------------------------------------------------
    def init_pose3d(self, override_segpose2d_load=False):
        self.load_pose2d_flag = True
        self.override_segpose2d_load = override_segpose2d_load

        if self.override_segpose2d_load == True:
            load_dir = self.segpose2d_dir if self.cfg.POSE3D.USE_SEGPOSE2D == True else self.pose2d_dir
        else:
            load_dir = self.pose2d_dir

        self.total_time_pose2d = len([file for file in os.listdir(os.path.join(load_dir, self.exo_camera_names[0], 'rgb')) if file.endswith('npy')])
        return

    def load_pose3d(self):
        self.load_pose3d_flag = True
        self.total_time_pose3d = len([file for file in os.listdir(self.pose3d_dir) if file.endswith('npy')])
        return         

    ##-------------------------------------------------------
    def init_pose2d(self):
        self.init_pose2d_rgb()
        self.init_pose2d_gray()
        return

    ##-------------------------------------------------------
    ## we assume pose2D vanilla is already done
    def init_segpose2d(self):
        # self.load_segmentation_flag = True ## loads the 2D poses from the vanilla pose2D for good initialization
        # self.total_time_segmentation = len([file for file in os.listdir(os.path.join(self.segmentation_dir, self.exo_camera_names[0], 'rgb')) if file.endswith('npy')])
        self.rgb_seg_pose_model = SegPoseModel(cfg=self.cfg, \
                                                pose_config=self.cfg.POSE2D.RGB_SEG_CONFIG_FILE, pose_checkpoint=self.cfg.POSE2D.RGB_SEG_CHECKPOINT)
        
        return

    def init_pose2d_rgb(self, dummy=False):
        ##------------------------2d pose model-----------------------
        if dummy == True:
            rgb_pose_config = self.cfg.POSE2D.DUMMY_RGB_CONFIG_FILE
            rgb_pose_checkpoint = self.cfg.POSE2D.DUMMY_RGB_CHECKPOINT
        else:
            rgb_pose_config = self.cfg.POSE2D.RGB_CONFIG_FILE
            rgb_pose_checkpoint = self.cfg.POSE2D.RGB_CHECKPOINT

        self.rgb_pose_model = PoseModel(cfg=self.cfg, pose_config=rgb_pose_config, pose_checkpoint=rgb_pose_checkpoint)

        return

    def init_segmentation(self):
        self.segmentation_model = SegmentationModel(cfg=self.cfg, model_type=self.cfg.SEGMENTATION.MODEL_TYPE, \
                                                    checkpoint=self.cfg.SEGMENTATION.CHECKPOINT, onnx_checkpoint=self.cfg.SEGMENTATION.ONNX_CHECKPOINT)
        
        ## the bounding box detector
        detector_config = self.cfg.POSE2D.DETECTOR_CONFIG_FILE
        detector_checkpoint = self.cfg.POSE2D.DETECTOR_CHECKPOINT
        self.detector_model = DetectorModel(cfg=self.cfg, detector_config=detector_config, detector_checkpoint=detector_checkpoint)

        return

    def init_bbox(self):
        if self.cfg.POSE2D.USE_BBOX_DETECTOR == True and self.load_pose2d_flag == False:
            detector_config = self.cfg.POSE2D.DETECTOR_CONFIG_FILE
            detector_checkpoint = self.cfg.POSE2D.DETECTOR_CHECKPOINT
            self.detector_model = DetectorModel(cfg=self.cfg, detector_config=detector_config, detector_checkpoint=detector_checkpoint)

        return

    def init_pose2d_gray(self):
        ##------------------------2d pose model-----------------------
        gray_pose_config = self.cfg.POSE2D.GRAY_CONFIG_FILE
        gray_pose_checkpoint = self.cfg.POSE2D.GRAY_CHECKPOINT
        self.gray_pose_model = PoseModel(cfg=self.cfg, pose_config=gray_pose_config, pose_checkpoint=gray_pose_checkpoint)
        return

    ##--------------------------------------------------------
    def update(self, time_stamp):
        self.time_stamp = time_stamp
        for aria_human_name in self.aria_humans.keys():
            self.aria_humans[aria_human_name].update(time_stamp=self.time_stamp)    

        for exo_camera_name in self.exo_cameras.keys():
            self.exo_cameras[exo_camera_name].update(time_stamp=self.time_stamp)        

        ## load 2d poses
        if self.load_pose2d_flag == True:
            self.pose2d = {}

            for camera_name, camera_mode in self.camera_names:

                if camera_mode == 'rgb':
                    if self.cfg.POSE3D.USE_SEGPOSE2D == True and self.override_segpose2d_load == False:
                        if camera_name in self.exo_camera_names:
                            pose2d_path = os.path.join(self.segpose2d_dir, camera_name, camera_mode, '{:05d}.npy'.format(time_stamp))
                            pose2d_results = np.load(pose2d_path, allow_pickle=True)
                        else:
                            # TODO: update to remove this loading!
                            pose2d_path = os.path.join(self.pose2d_dir, camera_name, camera_mode, '{:05d}.npy'.format(time_stamp))
                            pose2d_results = np.load(pose2d_path, allow_pickle=True)

                    else:
                        pose2d_path = os.path.join(self.pose2d_dir, camera_name, camera_mode, '{:05d}.npy'.format(time_stamp))
                        pose2d_results = np.load(pose2d_path, allow_pickle=True)

                    self.pose2d[(camera_name, camera_mode)] = pose2d_results

        ## load segmentation
        if self.load_segmentation_flag == True:
            self.segmentation = {}

            for camera_name, camera_mode in self.camera_names:
                if camera_mode == 'rgb' and camera_name in self.exo_camera_names:
                    segmentation_path = os.path.join(self.segmentation_dir, camera_name, camera_mode, '{:05d}.npz'.format(time_stamp))
                    segmentation_results = np.load(segmentation_path, allow_pickle=True)
                    segmentation_results = [segmentation_results[key] for key in segmentation_results][0][()] ## 0 for npz, next () for numpy array
                    self.segmentation[(camera_name, camera_mode)] = segmentation_results

        ## load 3d poses
        if self.load_pose3d_flag == True:

            if use.cfg.POSE3D.USE_SEGPOSE2D == True:
                pose3d_path = os.path.join(self.contact_pose3d_dir, '{:05d}.npy'.format(time_stamp))
            else:
                pose3d_path = os.path.join(self.pose3d_dir, '{:05d}.npy'.format(time_stamp))

            pose3d = np.load(pose3d_path, allow_pickle=True)
            self.set_poses3d(pose3d.item()) ## ndarray to dict, set the refined pose!

        ## load refined 3d poses
        if self.load_refine_pose3d_flag == True:
            pose3d_path = os.path.join(self.refine_pose3d_dir, '{:05d}.npy'.format(time_stamp))
            pose3d = np.load(pose3d_path, allow_pickle=True)
            self.set_poses3d(pose3d.item()) ## ndarray to dict, set the refined pose!

        ## load fitted 3d poses
        if self.load_fit_pose3d_flag == True:
            pose3d_path = os.path.join(self.fit_pose3d_dir, '{:05d}.npy'.format(time_stamp))
            pose3d = np.load(pose3d_path, allow_pickle=True)
            self.set_poses3d(pose3d.item())            

        ## load smpl params
        if self.load_smpl_flag == True:
            smpl_path = os.path.join(self.smpl_dir, '{:05d}.npy'.format(time_stamp))
            smpl = np.load(smpl_path, allow_pickle=True)
            self.set_smpl(smpl.item()) ## ndarray to dict      

         ## load init smpl params
        if self.load_init_smpl_flag == True:
            init_smpl_path = os.path.join(self.init_smpl_dir, '{:05d}.npy'.format(time_stamp))
            init_smpl = np.load(init_smpl_path, allow_pickle=True)
            self.set_smpl(init_smpl.item()) ## ndarray to dict            
        
        ## load smpl collision params
        if self.load_smpl_collision_flag == True:
            smpl_collision_path = os.path.join(self.smpl_collision_dir, '{:05d}.npy'.format(time_stamp))
            smpl_collision = np.load(smpl_collision_path, allow_pickle=True)
            self.set_smpl(smpl_collision.item())

        return

    ##--------------------------------------------------------
    def get_refine_poses3d(self):
        poses3d = {}
        for aria_human_name in self.aria_human_names:
            poses3d[aria_human_name] = self.aria_humans[aria_human_name].get_refine_poses3d()
        return poses3d

    def fit_poses3d(self):
        poses3d = {}
        for aria_human_name in self.aria_human_names:
            poses3d[aria_human_name] = self.aria_humans[aria_human_name].fit_poses3d()
        return poses3d

    ##--------------------------------------------------------
    def get_smpl(self):
        smpl_params = {}
        initial_smpls = self.load_initial_smpl()

        for human_name in initial_smpls.keys():
            smpl_params[human_name] = self.smplify.get_smpl(pose3d=self.aria_humans[human_name].pose3d, initial_smpl=initial_smpls[human_name])

        return smpl_params

    def get_smpl_trajectory(self):
        smpl_params_trajectory = {}
        initial_smpl_trajectory = self.load_initial_smpl_trajectory() ## H x T x smpl_info, nested dicts

        for human_name in self.aria_human_names:
            poses3d_trajectory = self.aria_humans[human_name].poses3d_trajectory ## T x 17 x 4
            smpl_params_trajectory[human_name] = self.smplify.get_smpl_trajectory(human_name=human_name, poses3d_trajectory=poses3d_trajectory, \
                    initial_smpl_trajectory=initial_smpl_trajectory[human_name])

        return smpl_params_trajectory
    
    def get_smpl_trajectory_collision(self):
        poses3d_trajectory_dict = {}
        smpls_trajectory_dict = {}

        for human_name in self.aria_human_names:
            poses3d_trajectory_dict[human_name] = self.aria_humans[human_name].poses3d_trajectory
            smpls_trajectory_dict[human_name] = self.aria_humans[human_name].smpls_trajectory

        smpl_params_trajectory = self.smplify.get_smpl_trajectory_collision(all_poses3d_trajectory=poses3d_trajectory_dict, all_initial_smpl_trajectory=smpls_trajectory_dict)

        return smpl_params_trajectory

    ##--------------------------------------------------------
    def load_initial_smpl(self, time_stamp=None):
        if time_stamp is None:
            time_stamp = self.time_stamp
        initial_smpl_path = os.path.join(self.init_smpl_dir, '{:05d}.npy'.format(time_stamp))
        initial_smpl = np.load(initial_smpl_path, allow_pickle=True)
        return initial_smpl.item()

    def load_initial_smpl_trajectory(self):
        initial_smpl_trajectory = {human_name:{} for human_name in self.aria_human_names} ## H x T x smpl
        time_stamps = list(range(1, self.total_time_fit_pose3d + 1))

        for t in time_stamps:
            smpl_trajectory = self.load_initial_smpl(time_stamp=t) ## human
            for human_name in smpl_trajectory.keys():
                if human_name in self.aria_human_names:
                    initial_smpl_trajectory[human_name][t] = smpl_trajectory[human_name]
        return initial_smpl_trajectory


    # ## initial estimate of the SMPL from CLIFF
    def get_initial_smpl(self, choosen_camera_names, bbox_padding=1.25):
        choosen_camera_names = [(camera_name, mode) for camera_name, mode in choosen_camera_names if camera_name in self.exo_cameras.keys()] ## filter by valid camera names
        choosen_cameras = [self.exo_cameras[camera_name] for camera_name, _ in choosen_camera_names]
        best_initial_smpl = {human_name: None for human_name in self.aria_human_names} ## dict by human names
        best_initial_smpl_error = {human_name: 1e5 for human_name in self.aria_human_names}

        for camera in choosen_cameras:
            bboxes = {}

            for human_name in self.aria_humans.keys():

                ## instead load the 3d pose and project to 2d
                pose3d = self.aria_humans[human_name].pose3d
                pose2d = camera.vec_project(pose3d[:, :3])

                ## check 2d pose inside the image, this will always be an exo image. So no worry about aria rotation!
                is_valid = (pose2d[:, 0] > 0) * (pose2d[:, 0] < camera.image_width) * (pose2d[:, 1] > 0) * (pose2d[:, 1] < camera.image_height)

                if is_valid.sum() >= 5:
                    pose2d = pose2d[is_valid]

                    x1 = pose2d[:, 0].min(); x2 = pose2d[:, 0].max()
                    y1 = pose2d[:, 1].min(); y2 = pose2d[:, 1].max()

                    bbox = np.array([x1, y1, x2, y2]) ## xyxy
                
                else:
                    bbox = camera.get_bbox_2d(aria_human=self.aria_humans[human_name]) ## xy, xy

                ## human not detected from this view
                if bbox is None:
                    continue

                bboxes[human_name] = bbox
        
            image_path = camera.get_image_path(time_stamp=self.time_stamp)
            initial_smpl = self.smpl_model.get_initial_smpl(image_path=image_path, bboxes=bboxes, bbox_padding=bbox_padding) ## dict by human name

            ## convert the smpl mesh to the global coordinate system
            for human_name in bboxes.keys():
                pose3d = self.aria_humans[human_name].pose3d

                all_pose2d = {}
                for exo_camera_name, exo_camera in self.exo_cameras.items():
                    pose2d = exo_camera.vec_project(pose3d[:, :3])

                    ## compute the 2d pose inside the image
                    pose2d_vis = (pose2d[:, 0] > 0) * (pose2d[:, 0] < exo_camera.image_width) * (pose2d[:, 1] > 0) * (pose2d[:, 1] < exo_camera.image_height)

                    ## concatentate the 2d pose
                    pose2d = np.concatenate([pose2d, pose2d_vis[:, None]], axis=1)
                    all_pose2d[exo_camera_name] = pose2d

                init_transl, init_global_orient, init_transformed_vertices, init_error = self.get_initial_smpl_transformation(camera, pose3d, initial_smpl[human_name], all_pose2d, self.exo_cameras)

                if init_error < best_initial_smpl_error[human_name]:
                    initial_smpl[human_name]['init_transl'] = init_transl
                    initial_smpl[human_name]['init_global_orient'] = init_global_orient
                    initial_smpl[human_name]['transformed_vertices'] = init_transformed_vertices ## the vertices transformed to the scene's global coordinate system
                    initial_smpl[human_name]['bbox'] = bboxes[human_name]
                    initial_smpl[human_name]['best_view'] = camera.camera_name

                    best_initial_smpl_error[human_name] = init_error
                    best_initial_smpl[human_name] = initial_smpl[human_name]

        for human_name in best_initial_smpl.keys():
            print('t:{} human:{} cam:{} best_init_error:{}'.format(self.time_stamp, human_name, best_initial_smpl[human_name]['best_view'], best_initial_smpl_error[human_name]))

        return best_initial_smpl

    ##-----------camera relative smpl to global coordinate best match------------------
    def get_initial_smpl_transformation(self, camera, pose3d, smpl, all_pose2d, exo_cameras, error_thres=1.0, \
                                        skip_face=False, limb_weight=1.0, distance_weight=1.0, height_weight=0.5):
        keypoints_coco = pose3d[:17, :3] ## 17 x 3
        init_joints = smpl['joints'] # 45 x 3, camera relative
        init_rotmat = smpl['rotmat'] ## 24 x 3 x 3; global + relative full body
        init_vertices = smpl['vertices'] 
        init_transl = smpl['cam_full'] ## 3
        init_pose = smpl['pose'] ## 72

        ###-------------------------run icp---------------------------------------------------
        init_joints_coco, init_mask = convert_kps(
                        init_joints.reshape(1, -1, 3),
                        mask=None,
                        src='smpl_45',
                        dst='coco'
                    )

        init_joints_coco = init_joints_coco[0] ## remove batch
        init_mask = init_mask == 1

        # ##----------------------------
        ## only consider common keypoints
        init_joints_coco = init_joints_coco[init_mask]
        keypoints_coco = keypoints_coco[init_mask]
        coco_keypoint_names = list(np.array(COCO_KP_ORDER)[init_mask])

        ##---recenter mesh's hip to 0,0,0. I dont know why this is working but it alignmnet is poor--
        ## my guess is that the smplx centers the mesh's hip to zero as well.
        init_joints_coco = init_joints_coco - init_joints[0] ## set the pelvis joint of the mesh to (0, 0, 0)
        init_vertices = init_vertices - init_joints[0] ## 

        mask = np.ones(len(init_joints_coco), dtype=np.bool)

        if skip_face == True:
            ## remove the face keypoints
            mask[coco_keypoint_names.index('nose')] = False
            mask[coco_keypoint_names.index('left_eye')] = False
            mask[coco_keypoint_names.index('right_eye')] = False
            mask[coco_keypoint_names.index('left_ear')] = False
            mask[coco_keypoint_names.index('right_ear')] = False
            init_transformation, distances, iterations = icp(init_joints_coco[mask], keypoints_coco[mask], max_iterations=20, tolerance=1e-6) ## 4 x 4
        else:
            init_transformation, distances, iterations = icp(init_joints_coco, keypoints_coco, max_iterations=20, tolerance=1e-6) ## 4 x 4

        ## transform the init_jointsco to the global coordinate system using init_transformation
        transformed_init_joints_coco = linear_transform(init_joints_coco, T=init_transformation)

        ## compute the limb lengths of the transformed init_joints_coco
        transformed_init_limb_lengths = compute_limb_length(transformed_init_joints_coco)
        keypoints_coco_limb_lengths = compute_limb_length(keypoints_coco)

        ## compute human height error, nose to center of left and right shoulder
        transformed_init_head_length = np.linalg.norm(transformed_init_joints_coco[coco_keypoint_names.index('nose')] \
                                                    - 0.5*(transformed_init_joints_coco[coco_keypoint_names.index('left_shoulder')] + transformed_init_joints_coco[coco_keypoint_names.index('right_shoulder')]))
        
        keypoints_coco_head_length = np.linalg.norm(keypoints_coco[coco_keypoint_names.index('nose')] \
                                                    - 0.5*(keypoints_coco[coco_keypoint_names.index('left_shoulder')] + keypoints_coco[coco_keypoint_names.index('right_shoulder')]))
        
        ## computer torso length error, center of left and right shoulder to center of left and right hip
        transformed_init_torso_length = np.linalg.norm(
            0.5*(transformed_init_joints_coco[coco_keypoint_names.index('left_shoulder')] + transformed_init_joints_coco[coco_keypoint_names.index('right_shoulder')]) \
                                                    - 0.5*(transformed_init_joints_coco[coco_keypoint_names.index('left_hip')] + transformed_init_joints_coco[coco_keypoint_names.index('right_hip')]))

        keypoints_coco_torso_length = np.linalg.norm(
            0.5*(keypoints_coco[coco_keypoint_names.index('left_shoulder')] + keypoints_coco[coco_keypoint_names.index('right_shoulder')]) \
                                                    - 0.5*(keypoints_coco[coco_keypoint_names.index('left_hip')] + keypoints_coco[coco_keypoint_names.index('right_hip')]))

        transformed_init_left_leg_length = np.linalg.norm(transformed_init_joints_coco[coco_keypoint_names.index('left_hip')] \
                                                          - transformed_init_joints_coco[coco_keypoint_names.index('left_knee')]) \
                                                + np.linalg.norm(transformed_init_joints_coco[coco_keypoint_names.index('left_knee')] \
                                                            - transformed_init_joints_coco[coco_keypoint_names.index('left_ankle')])
        keypoints_coco_left_leg_length = np.linalg.norm(keypoints_coco[coco_keypoint_names.index('left_hip')] \
                                                            - keypoints_coco[coco_keypoint_names.index('left_knee')]) \
                                                + np.linalg.norm(keypoints_coco[coco_keypoint_names.index('left_knee')] \
                                                            - keypoints_coco[coco_keypoint_names.index('left_ankle')])
        
        transformed_init_right_leg_length = np.linalg.norm(transformed_init_joints_coco[coco_keypoint_names.index('right_hip')] \
                                                            - transformed_init_joints_coco[coco_keypoint_names.index('right_knee')]) \
                                                + np.linalg.norm(transformed_init_joints_coco[coco_keypoint_names.index('right_knee')] \
                                                            - transformed_init_joints_coco[coco_keypoint_names.index('right_ankle')])
        keypoints_coco_right_leg_length = np.linalg.norm(keypoints_coco[coco_keypoint_names.index('right_hip')] \
                                                            - keypoints_coco[coco_keypoint_names.index('right_knee')]) \
                                                + np.linalg.norm(keypoints_coco[coco_keypoint_names.index('right_knee')] \
                                                            - keypoints_coco[coco_keypoint_names.index('right_ankle')])
        
        transformed_init_human_height = transformed_init_head_length + transformed_init_torso_length + transformed_init_left_leg_length + transformed_init_right_leg_length
        keypoints_coco_human_height = keypoints_coco_head_length + keypoints_coco_torso_length + keypoints_coco_left_leg_length + keypoints_coco_right_leg_length

        ## l2 norm of the difference between the transformed init_joints_coco and keypoints_coco, using the mask
        limb_error = np.mean(np.abs(transformed_init_limb_lengths - keypoints_coco_limb_lengths))
        distance_error = np.mean(mask * np.linalg.norm(transformed_init_joints_coco - keypoints_coco, axis=1))
        height_error = np.abs(transformed_init_human_height - keypoints_coco_human_height)

        icp_error = limb_weight*limb_error + distance_weight*distance_error + height_weight*height_error


        ## icp_error is the sum of 2dpose errors per view
        pose2d_error = 0
        for exo_camera_name, exo_camera in exo_cameras.items():
            pred_pose2d = exo_camera.vec_project(transformed_init_joints_coco)

            ## compute the 2d pose error
            pose2d = all_pose2d[exo_camera_name]

            pose2d_vis = pose2d[:, 2]
            pose2d = pose2d[:, :2]

            pose2d_error += np.mean(pose2d_vis * np.linalg.norm(pred_pose2d - pose2d, axis=1))

        icp_error = pose2d_error
        print('t:{} cam:{} icp_error:{} distance_error:{} limb_error:{} height_error:{}'.format(self.time_stamp, camera.camera_name, icp_error, distance_error, limb_error, height_error))

        ## remove the eyes and nose and ears
        if icp_error > error_thres:
            init_transformation, distances, iterations = icp(init_joints_coco, keypoints_coco, init_pose=init_transformation, max_iterations=20, tolerance=1e-6) ## 4 x 4

        ##------------------------------------------------------------------------------------
        init_transl = init_transformation[:3, 3].copy() - init_joints[0]## not exactly correct!
        init_global_orient_rotmat = init_transformation[:3, :3].copy() 
        init_global_orient = rotmat_to_aa(matrix=init_global_orient_rotmat)

        ###---------get how the verticies initialized would look like----------------
        transformed_vertices = self.smpl_model.get_initial_vertices(betas=smpl['betas'], \
                        body_pose_aa=smpl['pose'][3:], global_orient_aa=init_global_orient, transl=init_transl)
        
        return init_transl, init_global_orient, transformed_vertices, icp_error


    def save_initial_smpl(self, smpl, save_path):
        np.save(save_path, smpl, allow_pickle=True)
        return

    def save_smpl(self, smpl, save_path):
        np.save(save_path, smpl, allow_pickle=True)
        return


     ## no intersecting foot to the plane, currently deprecated
    def ground_plane_contact(self, vertices):
        max_offset = self.cfg.BLENDER.MAX_OFFSET
        tol = self.cfg.BLENDER.TOLERANCE
        mesh = trimesh.Trimesh(vertices, self.smpl_faces)

        signed_distance = self.proximity_manager.signed_distance(vertices) #https://trimsh.org/trimesh.proximity.html#trimesh.proximity.ProximityQuery.signed_distance
        intersecting_vertices = signed_distance > tol

        if intersecting_vertices.sum() > 0:
            distance_to_plane = min(max_offset, max(signed_distance[intersecting_vertices])) 
            normal = plane_unit_normal(self.ground_plane)
            vertices = vertices + distance_to_plane*normal

        return vertices

    def save_mesh_as_obj(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        all_vertices = {human_name: self.aria_humans[human_name].smpl['vertices'] for human_name in self.aria_human_names} ## 6890 x 3
        
        ##----refine to solve for the touching ground floor----
        # all_modified_vertices = self.ground_plane_contact(all_vertices) ## deprecated logic
        all_modified_vertices = all_vertices

        for human_name in all_modified_vertices.keys():
            human = self.aria_humans[human_name]
            vertices = all_modified_vertices[human_name]
            mesh = trimesh.Trimesh(vertices, self.smpl_faces)
            mesh.visual.face_colors = [human.color[2], human.color[1], human.color[0], 255*human.alpha] ## note the colors are bgr
            mesh.export(os.path.join(save_dir, 'mesh_{}.obj'.format(human_name)))

        return

    def save_mesh_as_obj_ego(self, save_dir, distance_thres=0.3):
        os.makedirs(save_dir, exist_ok=True)

        for human_name in self.aria_human_names:
            human = self.aria_humans[human_name]
            smpl_human = human.smpl
            vertices = smpl_human['vertices'] ## 6890 x 3
            mesh = trimesh.Trimesh(vertices, self.smpl_faces)

            ##--------save the aria location as obj-----------------
            transform = np.eye(4) ##4x4
            transform[:3, 3] = human.location ## place the sphere at the location
            head_mesh = trimesh.primitives.Sphere(radius=0.0001)
            head_mesh.apply_transform(transform)
            head_mesh.export(os.path.join(save_dir, 'mesh_head_{}.obj'.format(human_name)))

            ##------------------------------------------------------
            # ## delete the head
            if human_name == 'aria01':
                distances = np.sqrt(((vertices - human.location)**2).sum(axis=1))
                is_valid = distances < distance_thres
                head_vertices_idx = (is_valid.nonzero())[0]

                face_mask = (self.smpl_faces == self.smpl_faces)[:, 0] ## intialize

                for head_vertex_idx in head_vertices_idx:
                    is_vertex_in_faces = (self.smpl_faces == head_vertex_idx).sum(axis=1)
                    face_idxs = (is_vertex_in_faces.nonzero())[0]
                    face_mask[face_idxs] = False ## delete these faces

                mesh.update_faces(face_mask)

            mesh.visual.face_colors = [human.color[2], human.color[1], human.color[0], 255*human.alpha] ## note the colors are bg
            mesh.export(os.path.join(save_dir, 'mesh_{}.obj'.format(human_name)))

        return

    def get_ground_plane_mesh(self):

        prefix = ''

        if self.cfg.GEOMETRY.MANUAL_GROUND_PLANE_POINTS != '':
            prefix = 'manual_'

        ground_plane_mesh = pv.PolyData(self.scene_ground_vertices)
        volume = ground_plane_mesh.delaunay_3d(alpha=2.)

        shell = volume.extract_geometry()

        faces_as_array = shell.faces.reshape((-1, 4))[:, 1:]
        ground_mesh = trimesh.Trimesh(shell.points, faces_as_array)

        ground_mesh.export(os.path.join(self.colmap_dir, '{}raw_ground_plane_mesh.obj'.format(prefix)))

        print('ground plane saved to colmap dir! please import it in the scene.blend if not done already!')

        ##-----------save the ground plane equation as obj-----------------
        immutable_ground_plane = ground_mesh.bounding_box_oriented
        ground_plane = trimesh.Trimesh(vertices=immutable_ground_plane.vertices, faces=immutable_ground_plane.faces)
        ground_plane.export(os.path.join(self.colmap_dir, '{}ground_plane_mesh.obj'.format(prefix)))

        return ground_plane

    # #     colors = {
    #     'pink': np.array([197, 27, 125]),
    #     'light_pink': np.array([233, 163, 201]),
    #     'light_green': np.array([161, 215, 106]),
    #     'green': np.array([77, 146, 33]),
    #     'red': np.array([215, 48, 39]),
    #     'light_red': np.array([252, 146, 114]),
    #     'light_orange': np.array([252, 141, 89]),
    #     'purple': np.array([118, 42, 131]),
    #     'light_purple': np.array([175, 141, 195]),
    #     'light_blue': np.array([145, 191, 219]),
    #     'blue': np.array([69, 117, 180]),
    #     'gray': np.array([130, 130, 130]),
    #     'white': np.array([255, 255, 255]),
    #     'turkuaz': np.array([50, 134, 204]),
    # }

    def init_blender_vis(self):
        self.ground_plane_mesh = self.get_ground_plane_mesh()
        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('ground', self.ground_plane_mesh)
        self.proximity_manager = trimesh.proximity.ProximityQuery(self.ground_plane_mesh)

        ## the normal to the plane pointing upwards
        colmap_normal = plane_unit_normal(self.ground_plane) ## from the colmap
        trimesh_normals = self.ground_plane_mesh.face_normals
        
        ## figure out the trimesh normal clossest to the colmap normal
        normal_idx = ((colmap_normal*trimesh_normals).sum(axis=1)).argmax()
        self.ground_plane_normal = trimesh_normals[normal_idx]

        face = self.ground_plane_mesh.faces[normal_idx]
        self.ground_plane_origin = np.array(self.ground_plane_mesh.vertices[face[0]])

        return

    def blender_vis(self, mesh_dir, save_path):
        scene_name = self.cfg.BLENDER.SCENE_FILE
        colors = self.cfg.BLENDER.COLORS

        root_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', '..') 
        scene_file = os.path.join(root_dir, 'assets', scene_name) 

        blender_file = os.path.join(root_dir, 'lib', 'utils', 'blender.py')
        thickness = 0

        image_size = 1024
        focal_length = 2500
        output_file = save_path

        ####-------------------------------------------
        command = "blender -b {} \
          --python {} -- \
          -i {} \
          -o {} \
          -of {} \
          -t {} -f {} --sideview -c {} -s {}".format(\
                    scene_file, blender_file, \
                    mesh_dir, mesh_dir, output_file, thickness, focal_length,\
                    colors, image_size) 

        os.system(command)

        return

    def blender_vis_ego(self, aria_human_name, mesh_dir, save_path, scene_name='tagging/tagging_ego.blend'):
        ## comment or uncomment if scene is properly set
        # self.get_ground_plane_mesh()

        root_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', '..') 
        scene_file = os.path.join(root_dir, 'assets', scene_name) 

        ##---------------get camera translation and rotation-------------------
        ## world matrix of objects in blender
        blender_convention = np.zeros((4, 4))
        blender_convention[0, 0] = 1
        blender_convention[1, 1] = -4.371138828673793e-08
        blender_convention[1, 2] = -1
        blender_convention[2, 2] = -4.371138828673793e-08
        blender_convention[2, 1] = 1
        blender_convention[3, 3] = 1

        ###-------------------------------------------------------
        camera_rotation = self.cameras[(aria_human_name, 'rgb')].get_rotation()
        print('rotation', camera_rotation)
            
        ###--------------convert to blender camera system, xyz -> xzy
        camera_rotation_blender = camera_rotation
        camera_rotation_blender = R.from_matrix(camera_rotation_blender)
        camera_rotation_blender_euler = camera_rotation_blender.as_euler('xyz', degrees=False)
        camera_rotation_string = ':' +  ':'.join([str(val) for val in camera_rotation_blender_euler.tolist()]) + ':'

        blender_file = os.path.join(root_dir, 'lib', 'utils', 'blender_ego.py')
        thickness = 0
        colors = 'blue###green###red###orange'
        image_size = 1408
        focal_length = 2500
        output_file = save_path

        ####-------------------------------------------
        command = "blender -b {} \
          --python {} -- \
          -i {} \
          -o {} \
          -of {} \
          -t {} -f {} --sideview -c {} -s {} \
          --camera_rotation {}".format(\
                    scene_file, blender_file, \
                    mesh_dir, mesh_dir, output_file, thickness, focal_length,\
                    colors, image_size, camera_rotation_string) 
        os.system(command)

        return

    def draw_initial_smpl(self, smpl, save_path):
        original_image = self.view_camera.get_image(time_stamp=self.time_stamp)
        image = original_image.copy()
        overlay = 255*np.ones(image.shape)
        alpha = 0.7

        for human_name in smpl.keys():
            ## skip if human_name is the same as aria
            if human_name == self.viewer_name:
                continue
                
            smpl_human = smpl[human_name]
            color = self.aria_humans[human_name].color

            points_3d = smpl_human['transformed_vertices'] ## 6890 x 3
            # points_3d = self.aria_humans[human_name].pose3d[:, :3] 

            points_2d = self.view_camera.vec_project(points_3d)

            ## rotated poses from aria frame to human frame
            if self.view_camera.camera_type == 'ego':
                points_2d = self.view_camera.get_inverse_rotated_pose2d(pose2d=points_2d)

            is_valid = (points_2d[:, 0] >= 0) * (points_2d[:, 0] < image.shape[1]) * \
                        (points_2d[:, 1] >= 0) * (points_2d[:, 1] < image.shape[0])

            points_2d = points_2d[is_valid] ## only plot inside image points

            ## for exo
            if image.shape[0] > 1408:
                radius = 3

            else:
                radius = 1

            for idx in range(len(points_2d)):
                image = cv2.circle(image, (round(points_2d[idx, 0]), round(points_2d[idx, 1])), radius, color, -1)
                overlay = cv2.circle(overlay, (round(points_2d[idx, 0]), round(points_2d[idx, 1])), radius, color, -1)

        image = cv2.addWeighted(image, alpha, original_image, 1 - alpha, 0)
        image = np.concatenate([image, overlay], axis=1)

        ##----------------
        cv2.imwrite(save_path, image)

        return

    ##--------------------------------------------------------
    def draw_smpl(self, smpl, save_path):
        original_image = self.view_camera.get_image(time_stamp=self.time_stamp)
        image = original_image.copy()
        overlay = 255*np.ones(image.shape)
        alpha = 0.7

        for human_name in smpl.keys():
            ## skip if human_name is the same as aria
            if human_name == self.viewer_name:
                continue

            smpl_human = smpl[human_name]
            color = self.aria_humans[human_name].color

            points_3d = smpl_human['vertices']
            points_2d = self.view_camera.vec_project(points_3d)

            ## rotated poses from aria frame to human frame
            if self.view_camera.camera_type == 'ego':
                points_2d = self.view_camera.get_inverse_rotated_pose2d(pose2d=points_2d)

            is_valid = (points_2d[:, 0] >= 0) * (points_2d[:, 0] < image.shape[1]) * \
                        (points_2d[:, 1] >= 0) * (points_2d[:, 1] < image.shape[0])

            points_2d = points_2d[is_valid] ## only plot inside image points

            ## for exo
            if image.shape[0] > 1408:
                radius = 3
            else:
                radius = 1

            image, overlay = fast_circle(image, overlay, points_2d, radius, color)
            # image, overlay = slow_circle(image, overlay, points_2d, radius, color)

        image = cv2.addWeighted(image, alpha, original_image, 1 - alpha, 0)
        image = np.concatenate([image, overlay], axis=1)

        ##----------------
        cv2.imwrite(save_path, image)

        return

    ##--------------------------------------------------------
    def draw_smpl_pare_blender(self, smpl, mesh_save_dir, save_path):
        original_image = self.view_camera.read_undistorted_image(time_stamp=self.time_stamp)

        image = original_image.copy()
        overlay = 255*np.ones(image.shape)
        alpha = 0.7

        ##-----------ssave all objs---------------------
        for human_name in smpl.keys():
            ## skip if human_name is the same as aria
            if human_name == self.viewer_name:
                continue

            smpl_human = smpl[human_name]
            human = self.aria_humans[human_name]

            vertices = smpl_human['vertices']
            vertices_cam = linear_transform(points_3d=vertices, T=self.view_camera.extrinsics) ## vertices relative to the camera coordinate

            mesh = trimesh.Trimesh(vertices_cam, self.smpl_faces)
            mesh.visual.face_colors = [human.color[2], human.color[1], human.color[0], 255*human.alpha] ## note the colors are bgr

            # https://github.com/rawalkhirodkar/ochmr/blob/610f64bd04fdcaa3fa69c2f21f4b3551c6520916/pare/pare/utils/renderer.py
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh.export(os.path.join(mesh_save_dir, 'mesh_{}.obj'.format(human_name)))

        ##------------draw in blender-------------------
        ### diferent lighting setup for ego vs exo
        if self.view_camera.camera_name.startswith('aria'):
            scene_name = self.cfg.BLENDER.CAMERA_RELATIVE.EGO_SCENE_FILE
        else:
            scene_name = self.cfg.BLENDER.CAMERA_RELATIVE.SCENE_FILE
        colors = self.cfg.BLENDER.CAMERA_RELATIVE.COLORS

        root_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', '..') 
        scene_file = os.path.join(root_dir, 'assets', scene_name) 

        blender_file = os.path.join(root_dir, 'lib', 'utils', 'blender_camera_relative.py')
        thickness = 1.0

        image_width = original_image.shape[1]
        image_height = original_image.shape[0]

        K = self.view_camera.K ## assume undistort init is called for exo and ego

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        output_file = save_path

        if self.view_camera.camera_name.startswith('aria'):
            ## update the color list to remove self
            human_id = int(self.view_camera.camera_name.replace('aria', '')) - 1
            colors = colors.split('###')
            colors = [color for (idx, color) in enumerate(colors) if idx != human_id]
            colors = '###'.join(colors)

        ####-------------------------------------------
        command = "blender -b {} \
          --python {} -- \
          -i {} \
          -o {} \
          -of {} \
          -t {} -fx {} -fy {} -cx {} -cy {} -c {} -sx {} -sy {}".format(\
                    scene_file, blender_file, \
                    mesh_save_dir, mesh_save_dir, output_file, thickness, fx, fy, cx, cy,\
                    colors, image_width, image_height) 
        os.system(command)

        render_image = cv2.imread(output_file.replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)

        # ### rotate the render image if it is aria
        if self.view_camera.camera_name.startswith('aria'):
            render_image = cv2.rotate(render_image, cv2.ROTATE_90_CLOCKWISE)

        mask = render_image[:, :, -1]
        mask = 1.0*(mask > 0)
        mask = np.concatenate([mask.reshape(image.shape[0], image.shape[1], 1)]*3, axis=2)

        mask = cv2.GaussianBlur(mask, (5,5), cv2.BORDER_DEFAULT)

        render_image = render_image[:, :, :-1]
        image = mask*render_image + (1-mask)*image
        cv2.imwrite(save_path, image)

        ##--------delete the render-------------
        command = "rm -rf {}".format(output_file.replace('.jpg', '.png'))
        os.system(command)

        return

    ##--------------------------------------------------------
    def triangulate(self, flag='exo', secondary_flag='ego_rgb', debug=False, pose2d=None):
        if flag == 'ego':
            choosen_camera_names = [(camera_name, camera_mode) for (camera_name, camera_mode) in self.ego_camera_names_with_mode] ## only ego
        elif flag == 'exo':
            choosen_camera_names = [(camera_name, camera_mode) for (camera_name, camera_mode) in self.exo_camera_names_with_mode] ## only exo
        elif flag == 'ego_rgb':
            choosen_camera_names = [(camera_name, camera_mode) for (camera_name, camera_mode) in self.ego_camera_names_with_mode if camera_mode == 'rgb'] ## only ego, only rgb
        elif flag == 'all_rgb':
            choosen_camera_names = [(camera_name, camera_mode) for (camera_name, camera_mode) in self.exo_camera_names_with_mode] ## only exo
            choosen_camera_names += [(camera_name, camera_mode) for (camera_name, camera_mode) in self.ego_camera_names_with_mode if camera_mode == 'rgb'] ## exo + ego rgb
        elif flag == 'all':
            choosen_camera_names = [(camera_name, camera_mode) for (camera_name, camera_mode) in \
                                    self.ego_camera_names_with_mode + self.exo_camera_names_with_mode] ## both

        ##-------------------------------------------------------------
        if secondary_flag == 'ego_rgb':
            secondary_choosen_camera_names = [(camera_name, camera_mode) for (camera_name, camera_mode) in self.ego_camera_names_with_mode if camera_mode == 'rgb'] ## only ego, only rgb
        elif secondary_flag == None:
            secondary_choosen_camera_names = []

        ##-------------------------------------------------------------
        if debug == True:
            print('-------------------time_stamp:{}-------------------'.format(self.time_stamp))

        if pose2d is None:
            pose2d = self.pose2d

        triangulator = Triangulator(cfg=self.cfg, time_stamp=self.time_stamp, camera_names=choosen_camera_names, \
                        cameras={val: self.cameras[val] for val in choosen_camera_names}, \
                        secondary_camera_names=secondary_choosen_camera_names, \
                        secondary_cameras={val: self.cameras[val] for val in secondary_choosen_camera_names}, \
                        pose2d=pose2d, humans=self.aria_humans)

        poses3d = triangulator.run(debug=debug)
        # poses3d = triangulator.run_parallel(debug=debug)

        return poses3d

    def draw_poses3d(self, poses3d_list, save_path):
        ##---------------visualize---------------------------
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])

        for poses3d in poses3d_list:
            for human_name in poses3d.keys():
                points_3d = poses3d[human_name]
                color_string = self.aria_humans[human_name].color_string

                for _c in COCO_KP_CONNECTIONS:
                    ax.plot(xs=[points_3d[_c[0],0], points_3d[_c[1],0]], ys=[points_3d[_c[0],1], points_3d[_c[1],1]], zs=[points_3d[_c[0],2], points_3d[_c[1],2]], c=color_string)
                    # ax.scatter(xs=points_3d[:,0], ys=points_3d[:, 1], zs=points_3d[:, 2], c='blue')
                        
        plt.show()

        return

    def set_poses3d(self, poses3d):
        for human_name in poses3d.keys():
            if human_name not in self.aria_humans.keys():
                print('skipping {}'.format(human_name))
                continue
            self.aria_humans[human_name].set_pose3d(pose3d=poses3d[human_name])
        return

    def set_poses3d_trajectory(self, poses3d_trajectory):
        for human_name in poses3d_trajectory.keys():
            trajectory = poses3d_trajectory[human_name]
            trajectory = np.concatenate(trajectory, axis=0) ## T x 17 x 4
            self.aria_humans[human_name].set_poses3d_trajectory(poses3d_trajectory=trajectory)

        return

    def set_smpl(self, smpl):
        for human_name in smpl.keys():
            self.aria_humans[human_name].set_smpl(smpl=smpl[human_name])

        return

    def save_poses3d(self, poses3d, save_path):
        np.save(save_path, poses3d, allow_pickle=True)
        return

    ##--------------------------------------------------------
    def get_camera(self, camera_name='aria01', camera_mode='rgb'):
        camera = None
        view_type = None

        ## ego
        if camera_name in self.aria_human_names:
            view_type = 'ego'
            if camera_mode == 'rgb':
                camera = self.aria_humans[camera_name].rgb_cam

            elif camera_mode == 'left':
                camera = self.aria_humans[camera_name].left_cam

            elif camera_mode == 'right':
                camera = self.aria_humans[camera_name].right_cam

        elif camera_name in self.exo_camera_names:
            view_type = 'exo'
            camera = self.exo_cameras[camera_name]

        else:
            print('invalid camera name!: {},{}'.format(camera_name, camera_mode))
            exit()

        return camera, view_type

    ##--------------------------------------------------------
    def set_view(self, camera_name='aria01', camera_mode='rgb'):
        camera, view_type = self.get_camera(camera_name, camera_mode)
        self.view_camera = camera
        self.view_camera_type = camera_mode
        self.view_type = view_type ## ego or exo
        self.viewer_name = camera_name

        return 

    ##-----------------------bboxes----------------------------
    def get_bboxes(self):
        bboxes = []
        aria_humans = [aria_human for aria_human_name, aria_human in self.aria_humans.items() if aria_human_name != self.viewer_name]

        for aria_human in aria_humans:
            bbox = self.view_camera.get_bbox_2d(aria_human=aria_human)

            if bbox is not None:
                bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 1]) ## add confidnece

                bboxes.append({'bbox': bbox, \
                                'human_name': aria_human.human_name, \
                                'human_id': aria_human.human_id, \
                                'color': aria_human.color})

        if self.cfg.POSE2D.USE_BBOX_DETECTOR == True:
            image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)
            bboxes, offshelf_bboxes = self.detector_model.get_bboxes(image_name=image_name, bboxes=bboxes)

            if self.cfg.BBOX.SAVE_OFFSHELF_BOX_TO_DISK == True:
                cam_dir = os.path.join(self.offshelf_bbox_dir, self.view_camera.camera_name)
                os.makedirs(cam_dir, exist_ok=True)
                
                ## save offshelf bboxes, 05d.pkl (frame_id)
                offshelf_bboxes_path = os.path.join(cam_dir, '{:05d}.pkl'.format(self.time_stamp))
                with open(offshelf_bboxes_path, 'wb') as f:
                    pickle.dump(offshelf_bboxes, f)

        return bboxes
    
    # ## this function only runs sequentially
    def get_head_bboxes(self):
        head_bboxes = []
        aria_humans = [aria_human for aria_human_name, aria_human in self.aria_humans.items() if aria_human_name != self.viewer_name]

        for aria_human in aria_humans:
            head_bbox = self.view_camera.get_head_bbox_2d(aria_human=aria_human)
            distance_to_camera = np.linalg.norm(aria_human.location - self.view_camera.location)

            ## round to 2 decimal places
            distance_to_camera = np.round(distance_to_camera, 2)

            if head_bbox is not None:
                head_bbox = np.array([head_bbox[0], head_bbox[1], head_bbox[2], head_bbox[3], 1])

                head_bboxes.append({'head_bbox': head_bbox, \
                                'distance_to_camera': distance_to_camera, \
                                'human_name': aria_human.human_name, \
                                'human_id': aria_human.human_id, \
                                'color': aria_human.color})
        
        return head_bboxes

    ## fasterrcnn offshelf bboxes saved to disk
    def get_offshelf_bboxes(self):
        offshelf_bboxes_path = os.path.join(self.offshelf_bbox_dir, self.view_camera.camera_name, '{:05d}.pkl'.format(self.time_stamp))
        with open(offshelf_bboxes_path, 'rb') as f:
            bboxes = pickle.load(f)

        return bboxes

    def save_bboxes(self, bboxes, save_path):
        np.save(save_path, bboxes, allow_pickle=True)
        return

    def save_segmentation(self, segmentation, save_path, compress=False):
        
        ## compress the numpy array segmentation using numpy.savez_compressed
        if compress == True:
            np.savez_compressed(save_path.replace('.npy', ''), segmentation)
        else:
            np.save(save_path, segmentation, allow_pickle=True)

        return

    def load_bboxes(self):
        bboxes_path = os.path.join(self.bbox_dir, self.view_camera.camera_name, self.view_camera.type_string, '{:05d}.npy'.format(self.time_stamp))
        bboxes = np.load(bboxes_path, allow_pickle=True).tolist()
        return bboxes

    def load_segmentation(self):
        segmentation_path = os.path.join(self.segmentation_dir, self.view_camera.camera_name, self.view_camera.type_string, '{:05d}.npz'.format(self.time_stamp))
        segmentation_results = np.load(segmentation_path, allow_pickle=True)
        segmentation_results = [segmentation_results[key] for key in segmentation_results][0][()] ## 0 for npz, next () for numpy array
        return segmentation_results

    def draw_bboxes(self, bboxes, save_path):
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)
        image = cv2.imread(image_name)

        for bbox_2d_info in bboxes:
            bbox_2d = bbox_2d_info['bbox']
            color = bbox_2d_info['color']

            if image.shape[0] > 1408:
                thickness = 12
            elif image.shape[0] == 1408:
                thickness = 5
            else:
                thickness = 2

            image = cv2.rectangle(image, (round(bbox_2d[0]), round(bbox_2d[1])), (round(bbox_2d[2]), round(bbox_2d[3])), color, thickness)
        
        cv2.imwrite(save_path, image)

        return 

    def draw_scene_vertices(self, save_path):
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)
        image = cv2.imread(image_name)

        points_2d = self.view_camera.vec_project(self.scene_ground_vertices) ## N x 2

        if self.view_camera.camera_type == 'ego':
            x = points_2d[:, 0].copy()
            y = points_2d[:, 1].copy()

            rotated_x = self.view_camera.rotated_image_height - y
            rotated_y = x 

            points_2d[:, 0] = rotated_x
            points_2d[:, 1] = rotated_y
        
        ## for exo
        if image.shape[0] > 1408:
            radius = 3
        else:
            radius = 1

        for idx in range(len(points_2d)):
            image = cv2.circle(image, (round(points_2d[idx, 0]), round(points_2d[idx, 1])), radius, [255, 255, 0], -1)
        
        cv2.imwrite(save_path, image)

        return 

    ##-----------------------get aria locations------------------
    def get_aria_locations(self, debug=False):
        aria_locations = []
        aria_humans = [aria_human for aria_human_name, aria_human in self.aria_humans.items() if aria_human_name != self.viewer_name]

        for aria_human in aria_humans:
            aria_human_location_2d, is_valid = self.view_camera.get_aria_location(point_3d=aria_human.location)

            if debug == True:
                print('view: {}, human:{}, loc:{}, is_valid:{}'.format(self.view_camera.camera_name, aria_human.human_name, aria_human_location_2d, is_valid))

            ## inside the frame            
            if is_valid:
                aria_locations.append({'location':aria_human_location_2d, 'color': aria_human.color, 'human_name': aria_human.human_name})

        return aria_locations

    def get_exo_locations(self):
        exo_locations = []

        for exo_camera_name, exo_camera in self.exo_cameras.items():
            location_2d, is_valid = self.view_camera.get_aria_location(point_3d=exo_camera.location)

            ## inside the frame            
            if is_valid:
                exo_locations.append({'location':location_2d, 'color': [255, 255, 0], 'human_name': exo_camera_name})

        return exo_locations

    def draw_camera_locations(self, aria_locations_2d, save_path):
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)
        image = cv2.imread(image_name)

        for location_info in aria_locations_2d:
            location = location_info['location']
            color = location_info['color']

            ## for exo
            if image.shape[0] > 1408:
                radius = 10
            elif image.shape[0] < 1408:
                radius = 3
            else:
                radius = 5

            image = cv2.circle(image, (round(location[0]), round(location[1])), radius, color, -1)
        
        cv2.imwrite(save_path, image)

        return 

    ## both ego and exo
    def get_camera_locations(self):
        aria_locations = self.get_aria_locations()
        exo_locations = self.get_exo_locations()

        locations = aria_locations + exo_locations

        return locations


    ##-------------------------poses2d---------------------------
    def get_poses2d(self):
        bboxes = self.load_bboxes()
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)

        if self.view_camera.type_string == 'rgb':
            pose_results = self.rgb_pose_model.get_poses2d(bboxes=bboxes, \
                                        image_name=image_name, \
                                        camera_type=self.view_camera.camera_type, ## ego or exo
                                        camera_mode=self.view_camera.type_string,
                                        camera=self.view_camera,
                                        aria_humans=self.aria_humans,
                                    )
        else:
            pose_results = self.gray_pose_model.get_poses2d(bboxes=bboxes, \
                                        image_name=image_name, \
                                        camera_type=self.view_camera.camera_type, ## ego or exo
                                        camera_mode=self.view_camera.type_string,
                                        camera=self.view_camera,
                                        aria_humans=self.aria_humans,
                                    )

        return pose_results 
    
    ##-------------------------seg poses2d---------------------------
    def get_segposes2d(self, segmentations=None):
        
        if segmentations == None:
            segmentations = self.segmentation[(self.view_camera.camera_name, 'rgb')] ## dict human: info
        
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)

        assert self.view_camera.type_string == 'rgb', 'seg poses2d only works for rgb camera'
        pose_results = self.rgb_seg_pose_model.get_poses2d(segmentations=segmentations, \
                                    image_name=image_name, \
                                    camera_type=self.view_camera.camera_type, ## ego or exo
                                    camera_mode=self.view_camera.type_string,
                                    camera=self.view_camera,
                                    aria_humans=self.aria_humans,
                                )

        return pose_results 

    def get_segmentation(self):
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)
        poses2d = self.get_projected_poses3d() ## because we set the predicted_poses3d by KF before
        segmentation = self.segmentation_model.get_segmentation(image_name=image_name, poses2d=poses2d) # #dict human: segmentation

        ## add color for visualization
        for human_name in segmentation.keys():
            segmentation[human_name]['color'] = self.aria_humans[human_name].color

        return segmentation

    def get_segmentation_association(self):
        head_bboxes = self.get_head_bboxes()
        prev_segmentation = None

        ## load the previous segmentation which is correctly associated
        if self.time_stamp != 1:
            prev_segmentation_path = os.path.join(self.segmentation_dir, self.view_camera.camera_name, \
                                                  self.view_camera.type_string, '{:05d}.npz'.format(self.time_stamp - 1))
            prev_segmentation = np.load(prev_segmentation_path, allow_pickle=True)
            prev_segmentation = [prev_segmentation[key] for key in prev_segmentation][0][()]

        ## load the current segmentation which is not associated
        offshelf_segmentation_path = os.path.join(self.offshelf_segmentation_dir, self.view_camera.camera_name, \
                                                    self.view_camera.type_string, '{:05d}.npz'.format(self.time_stamp))

        ## load the .npz file
        offshelf_segmentation = np.load(offshelf_segmentation_path, allow_pickle=True)
        offshelf_segmentation = [offshelf_segmentation[key] for key in offshelf_segmentation][0] ## B x H x W

        ## convert array to list
        offshelf_segmentation = [offshelf_segmentation[i] for i in range(offshelf_segmentation.shape[0])]

        ## get the associated segmentation
        segmentation = self.segmentation_model.get_segmentation_association(head_bboxes=head_bboxes, \
                                                                            prev_segmentation=prev_segmentation, \
                                                                            segmentation=offshelf_segmentation, \
                                        )
        return segmentation
    
    def draw_segmentation(self, segmentation, save_path):
        image = self.view_camera.get_image(time_stamp=self.time_stamp)
        mask_image = self.segmentation_model.draw_segmentation(segmentation, image)
        cv2.imwrite(save_path, mask_image)
        return

    def save_poses2d(self, pose_results, save_path):
        np.save(save_path, pose_results, allow_pickle=True)
        return   

    def draw_poses2d(self, pose_results, save_path):
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)

        if self.view_camera.type_string == 'rgb':

            ## if rgb_seg_pose_model exists, use it
            ## check if the attribute exists
            if hasattr(self, 'rgb_seg_pose_model') and self.rgb_seg_pose_model is not None:
                self.rgb_seg_pose_model.draw_poses2d(pose_results, image_name, save_path, \
                                    camera_type=self.view_camera.camera_type, ## ego or exo
                                    camera_mode=self.view_camera.type_string)
            else:
                self.rgb_pose_model.draw_poses2d(pose_results, image_name, save_path, \
                                        camera_type=self.view_camera.camera_type, ## ego or exo
                                        camera_mode=self.view_camera.type_string)

        else:
            self.gray_pose_model.draw_poses2d(pose_results, image_name, save_path, \
                                    camera_type=self.view_camera.camera_type, ## ego or exo
                                    camera_mode=self.view_camera.type_string)

        return    

    def draw_rotated_poses2d(self, pose_results, save_path):
        image_name = self.view_camera.get_rotated_image_path(time_stamp=self.time_stamp)

        for idx in range(len(pose_results)):
            keypoints = pose_results[idx]['keypoints']
            bbox = pose_results[idx]['bbox']
            pose_results[idx]['keypoints'], pose_results[idx]['bbox'] = self.view_camera.get_rotated_pose2d(pose2d=keypoints, bbox=bbox)

        if self.view_camera.type_string == 'rgb':
            self.rgb_pose_model.draw_poses2d(pose_results, image_name, save_path, \
                                    camera_type=self.view_camera.camera_type, ## ego or exo
                                    camera_mode=self.view_camera.type_string)

        else:
            self.gray_pose_model.draw_poses2d(pose_results, image_name, save_path, \
                                    camera_type=self.view_camera.camera_type, ## ego or exo
                                    camera_mode=self.view_camera.type_string)

        return    

    ## only of the view of the scene
    def load_poses2d(self):
        pose2d_path = os.path.join(self.pose2d_dir, self.view_camera.camera_name, self.view_camera.type_string, '{:05d}.npy'.format(self.time_stamp))
        pose2d = np.load(pose2d_path, allow_pickle=True).tolist()   
        return pose2d

    ##-------------------------------------------------------------
    def get_poses3d(self):
        return {human_name: self.aria_humans[human_name].pose3d for human_name in self.aria_humans}

    ##-----------------project the pose3d of the aria humans from the viewer-------
    def get_projected_poses3d(self):
        pose_results = {}

        for human_name in self.aria_humans:
            if human_name != self.viewer_name:
                pose3d = self.aria_humans[human_name].pose3d ## 17 x 4
                projected_pose3d = self.view_camera.vec_project(pose3d[:, :3]) ## 17 x 2
                projected_pose3d = np.concatenate([projected_pose3d, pose3d[:, 3].reshape(-1, 1)], axis=1) ## 17 x 3

                ## rotated poses from aria frame to human frame
                if self.view_camera.camera_type == 'ego':
                    projected_pose3d = self.view_camera.get_inverse_rotated_pose2d(pose2d=projected_pose3d)

                pose_results[human_name] = projected_pose3d

        return pose_results

    ## 3d poses in camera coordinate system, RGB
    def get_cam_poses3d(self):
        assert self.viewer_name.startswith('aria')
        pose_results = {}

        for human_name in self.aria_humans:
            if human_name != self.viewer_name:
                pose3d = self.aria_humans[human_name].pose3d ## 17 x 4
                pose3d_cam = self.view_camera.vec_cam_from_world(pose3d[:, :3])
                pose3d_cam = np.concatenate([pose3d_cam, pose3d[:, 3].reshape(-1, 1)], axis=1) ## 17 x 4, note its in the rotated camera

                pose_results[human_name] = pose3d_cam

        return pose_results

    def draw_projected_poses3d(self, pose_results, save_path):
        image_name = self.view_camera.get_image_path(time_stamp=self.time_stamp)

        ##------------this is a generic function------------------
        ## check if self.rgb_pose_model attribute to the class exists
        if hasattr(self, 'rgb_seg_pose_model'):
            self.rgb_seg_pose_model.draw_projected_poses3d(pose_results, image_name, save_path, \
                                camera_type=self.view_camera.camera_type, ## ego or exo
                                camera_mode=self.view_camera.type_string)
            
        else:
            self.rgb_pose_model.draw_projected_poses3d(pose_results, image_name, save_path, \
                                        camera_type=self.view_camera.camera_type, ## ego or exo
                                        camera_mode=self.view_camera.type_string)

        return    

    ##--------------------------------------------------------
    def get_image(self):
        return self.view_camera.get_image(time_stamp=self.time_stamp) ## opencv, BGR image

    ##--------------------------------------------------------
    def debug(self):
        radius = 1
        scene_list = []

        ##-------ego spheres-----------
        for aria_human_name in self.aria_human_names:
            aria_human = self.aria_humans[aria_human_name]
            sphere = aria_human.get_sphere_mesh(point_3d=aria_human.location, radius=radius)
            scene_list.append(sphere)

        ##-------exo spheres-----------
        for exo_camera_name in self.exo_camera_names:
            exo_camera = self.exo_cameras[exo_camera_name]
            sphere = exo_camera.get_sphere_mesh(point_3d=exo_camera.location, radius=radius)
            scene_list.append(sphere)

        scene = trimesh.Scene(scene_list)
        scene.show()

        return

        ego_objects = [self.aria_humans[aria_human_name] for aria_human_name in self.aria_human_names]
        run_debug(ego_objects)

        return

    ##-----------------------------------------------------------------
    def get_colmap_camera_mapping(self):
        self.intrinsics_calibration_file = os.path.join(self.colmap_dir, 'cameras.txt')
        self.extrinsics_calibration_file = os.path.join(self.colmap_dir, 'images.txt')

        with open(self.intrinsics_calibration_file) as f:
            intrinsics = f.readlines()
            intrinsics = intrinsics[3:] ## drop the first 3 lines

        colmap_camera_ids = []
        is_exo_camera = []
        for line in intrinsics:
            line = line.split()
            colmap_camera_id = int(line[0])
            colmap_camera_model = line[1]
            image_width = int(line[2])
            image_height = int(line[3])

            colmap_camera_ids.append(colmap_camera_id)

            if image_height == 1408 and image_width == 1408:
                is_exo_camera.append(False)
            else:
                is_exo_camera.append(True)

        num_colmap_arias = len(is_exo_camera) - sum(is_exo_camera)
        num_arias = len(os.listdir(self.ego_dir))
        exo_camera_names = sorted(os.listdir(self.exo_dir))

        # assert(num_colmap_arias == num_arias)

        ## get the name of the folder containing the camera name for the exo cameras
        exo_camera_mapping = {}
        for (colmap_camera_id, is_valid) in zip(colmap_camera_ids, is_exo_camera):
            if is_valid == True:
                exo_camera_name = self.get_camera_name_from_colmap_camera_id(colmap_camera_id)
                assert(exo_camera_name is not None)
                exo_camera_mapping[exo_camera_name] = colmap_camera_id            

        return exo_camera_mapping

    ##--------------------------------------------------------
    def get_camera_name_from_colmap_camera_id(self, colmap_camera_id):
        with open(self.extrinsics_calibration_file) as f:
            extrinsics = f.readlines()
            extrinsics = extrinsics[4:] ## drop the first 4 lines
            extrinsics = extrinsics[::2] ## only alternate lines

        for line in extrinsics:
            line = line.strip().split()
            camera_id = int(line[-2])
            image_path = line[-1]
            camera_name = image_path.split('/')[0]

            if camera_id == colmap_camera_id:
                return camera_name

        return None