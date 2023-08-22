import numpy as np
import os
import cv2
from tqdm import tqdm
import sys
import pathlib
import torch
import mmcv
from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
from mmhuman3d.core.evaluation import keypoint_mpjpe
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mmhuman3d.data.data_structures import HumanData
from mmhuman3d.models.registrants.builder import build_registrant
from mmhuman3d.core.conventions.keypoints_mapping import KEYPOINTS_FACTORY
from mmhuman3d.utils.transforms import rotmat_to_aa
from mmhuman3d.core.conventions.keypoints_mapping.coco import COCO_KEYPOINTS
from mmhuman3d.core.conventions.keypoints_mapping.smpl import SMPL_45_KEYPOINTS
from collections import Counter
from utils.icp import icp
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
mmhuman3d_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', '..', '..', 'mmhuman3d') 

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

##------------------------------------------------------------------------------------
class SMPLify:
    def __init__(self, cfg):
        self.cfg = cfg
        self.keypoint_type = 'coco' ## 17 keypoints
        self.input_type = 'keypoints3d'

        self.config_file = os.path.join(mmhuman3d_dir, 'configs', 'smplify', cfg.SMPL.CONFIG_FILE)

        self.body_model_dir = os.path.join(mmhuman3d_dir, 'data', 'body_models')

        self.device = torch.device('cuda')
        self.num_betas = 10
        self.batch_size = 1

        self.original_smplify_config = mmcv.Config.fromfile(self.config_file)
        assert self.original_smplify_config.body_model.type.lower() in ['smpl', 'smplx']
        assert self.original_smplify_config.type.lower() in ['smplify', 'smplifyx']


        self.collision_config_file = os.path.join(mmhuman3d_dir, 'configs', 'smplify', cfg.SMPL_COLLISION.CONFIG_FILE)
        self.original_smplify_collision_config = mmcv.Config.fromfile(self.collision_config_file)

        return


    def build_smplify(self, human_name):
        all_human_names = self.cfg.SMPL.ARIA_NAME_LIST
        idx = all_human_names.index(human_name)
        num_epochs = self.cfg.SMPL.NUM_EPOCHS_LIST[idx]
        stage1_iters = self.cfg.SMPL.STAGE1_ITERS_LIST[idx]
        stage2_iters = self.cfg.SMPL.STAGE2_ITERS_LIST[idx]
        stage3_iters = self.cfg.SMPL.STAGE3_ITERS_LIST[idx]
        gender = self.cfg.SMPL.ARIA_GENDER_LIST[idx]

        # create body model
        self.body_model_config = dict(
                type=self.original_smplify_config.body_model.type.lower(),
                gender=gender,
                num_betas=self.num_betas,
                model_path=self.body_model_dir,
                batch_size=self.batch_size,
            )

        ##---------build-------------
        smplify_config = self.original_smplify_config.copy()
        smplify_config.update(dict(
                            verbose=self.cfg.SMPL.VERBOSE,
                            body_model=self.body_model_config,
                            use_one_betas_per_video=True,
                            num_epochs=num_epochs))
        smplify_config['stages'][0]['num_iter'] = stage1_iters
        smplify_config['stages'][1]['num_iter'] = stage2_iters
        smplify_config['stages'][2]['num_iter'] = stage3_iters

        smplify = build_registrant(dict(smplify_config))

        return smplify, smplify_config
    
    def build_smplify_collision(self, human_names, gender='neutral'):
        num_epochs = self.cfg.SMPL_COLLISION.NUM_EPOCHS

        # create body model
        self.body_model_collision_config = dict(
                type=self.original_smplify_collision_config.body_model.type.lower(),
                gender=gender,
                num_betas=self.num_betas,
                model_path=self.body_model_dir,
                batch_size=self.batch_size,
            )

        ##---------build-------------
        smplify_collision_config = self.original_smplify_collision_config.copy()
        smplify_collision_config.update(dict(
                            verbose=True,
                            body_model=self.body_model_collision_config,
                            use_one_betas_per_video=True,
                            num_epochs=num_epochs))

        smplify_collision = build_registrant(dict(smplify_collision_config))

        return smplify_collision, smplify_collision_config

    def get_smpl_trajectory(self, human_name, poses3d_trajectory, initial_smpl_trajectory, skip_face=False):
        """
        all_poses3d_trajectory is time x 17 x 4
        initial_smpl_trajectory is time x smpl_info --> nested dicts
        """
        assert(poses3d_trajectory.shape[0] == len(initial_smpl_trajectory.keys()))
        assert(poses3d_trajectory.shape[1] == 17)
        assert(poses3d_trajectory.shape[2] == 4)

        self.smplify, self.smplify_config = self.build_smplify(human_name)

        total_time = poses3d_trajectory.shape[0]
        keypoints_src = poses3d_trajectory[:, :17, :3] ## take the first 17 keypoints, T x 17 x 3

        src_mask = np.ones(17)

        if skip_face == True:
            ## set weight of face keypoints to 0
            src_mask[COCO_KEYPOINTS.index('nose')] = 0
            src_mask[COCO_KEYPOINTS.index('left_eye')] = 0
            src_mask[COCO_KEYPOINTS.index('right_eye')] = 0
            src_mask[COCO_KEYPOINTS.index('left_ear')] = 0
            src_mask[COCO_KEYPOINTS.index('right_ear')] = 0
        
        if human_name not in self.cfg.SMPL.JOINT_WEIGHT_OVERRIDE.ARIA_NAME_LIST:
            keypoints, mask =  convert_kps(
                                keypoints_src,
                                mask=src_mask,
                                src=self.keypoint_type,
                                dst=self.smplify_config.body_model['keypoint_dst']
                            )
        else:
            idx = self.cfg.SMPL.JOINT_WEIGHT_OVERRIDE.ARIA_NAME_LIST.index(human_name)
            override_joint_names = self.cfg.SMPL.JOINT_WEIGHT_OVERRIDE.JOINT_NAMES[idx]
            override_joint_weights = self.cfg.SMPL.JOINT_WEIGHT_OVERRIDE.JOINT_WEIGHTS[idx]

            ## update the src_mask
            for joint_name, joint_weight in zip(override_joint_names, override_joint_weights):
                joint_idx = COCO_KEYPOINTS.index(joint_name)
                src_mask[joint_idx] = joint_weight

            keypoints, mask =  convert_kps(
                                keypoints_src,
                                mask=src_mask,
                                src=self.keypoint_type,
                                dst=self.smplify_config.body_model['keypoint_dst']
                            )

        keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

        ##---------------beta----------------------
        # dict_keys(['betas', 'rotmat', 'pose', 'cam_full', 'vertices', 'joints', 'focal_length', 'init_transl', 'init_global_orient', 'transformed_vertices', 'bbox', 'best_view'])
        beta = []
        for t in initial_smpl_trajectory.keys():
            beta.append(initial_smpl_trajectory[t]['betas'].reshape(1, -1)) ## 1 x 10
        beta = np.concatenate(beta, axis=0) ## t x 10

        ##---------------init_body_pose----------------------
        init_body_pose = []
        for t in initial_smpl_trajectory.keys():
            body_pose = initial_smpl_trajectory[t]['pose'].reshape(-1, 3) ## 24 x 3
            body_pose = body_pose[1:] ## drop the global pose
            body_pose = body_pose.reshape(1, -1) ## 1 x 69
            init_body_pose.append(body_pose) 
        init_body_pose = np.concatenate(init_body_pose, axis=0) ## t x 69

        ##-----------------init_transl-------------------
        init_transl = []
        for t in initial_smpl_trajectory.keys():
            init_transl.append(initial_smpl_trajectory[t]['init_transl'].reshape(1, -1)) ## 1 x 3
        init_transl = np.concatenate(init_transl, axis=0) ## t x 3

        ##-----------------init_global_orient-------------------
        init_global_orient = []
        for t in initial_smpl_trajectory.keys():
            init_global_orient.append(initial_smpl_trajectory[t]['init_global_orient'].reshape(1, -1)) ## 1 x 3
        init_global_orient = np.concatenate(init_global_orient, axis=0) ## t x 3


        ##---------------------post processing------------------------------------------
        ## cluster the beta (shape parameters) into different clusters
        beta_centroid = self.cluster_beta(beta)
        raw_beta = beta.copy()

        ## replace the beta with the centroid
        beta = beta_centroid.repeat(total_time, axis=0)

        ##-------------------------------------------------------------------------------
        keypoints = torch.tensor(keypoints, dtype=torch.float32, device=self.device)
        keypoints_conf = torch.tensor(keypoints_conf, dtype=torch.float32, device=self.device)

        init_beta = torch.tensor(beta, dtype=torch.float32, device=self.device)
        init_body_pose = torch.tensor(init_body_pose, dtype=torch.float32, device=self.device)
        init_transl = torch.tensor(init_transl, dtype=torch.float32, device=self.device)
        init_global_orient = torch.tensor(init_global_orient, dtype=torch.float32, device=self.device)

        ## initial with the CLIFF predictions, throw away the global information
        human_data = dict(
                            human_name=human_name, \
                            keypoints3d=keypoints, \
                            keypoints3d_conf=keypoints_conf, \
                            init_betas=init_beta, \
                            init_body_pose=init_body_pose, \
                            init_transl=init_transl, \
                            init_global_orient=init_global_orient,\
                        )

        # run SMPLify(X)
        smplify_output, smplify_output_per_epoch = self.smplify(**human_data, return_joints=True, return_verts=True)

        ret = {t:{} for t in range(total_time)}

        for key in smplify_output.keys():
            for t in range(total_time):
                if key == 'epoch_loss':
                    ret[t][key] = smplify_output[key]
                elif key == 'betas':
                    ret[t][key] = (smplify_output[key][0].cpu().numpy()).reshape(-1)
                else:
                    ret[t][key] = smplify_output[key][t].cpu().numpy() 

        return ret
    
    def cluster_beta(self, beta, min_cluster_size=3):
        # Standardize features to zero mean and unit variance
        scaler = StandardScaler()
        beta_scaled = scaler.fit_transform(beta)

        # Create an HDBSCAN instance and fit the data
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size)) # Adjust this value based on your data
        clusterer.fit(beta_scaled)

        labels = clusterer.labels_  # Cluster labels for each point in the dataset (-1 indicates noise).
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # Number of clusters in labels, ignoring noise if present.

        print(f'Estimated number of clusters: {n_clusters_}')

        # Count the number of occurrences of each label (excluding noise points)
        label_counts = Counter(label for label in labels if label != -1)

        # Find the label of the largest cluster
        largest_cluster_label = max(label_counts.keys(), key=label_counts.get)

        # Get all points in the largest cluster
        largest_cluster_points = beta_scaled[labels == largest_cluster_label]

        # Compute the centroid of the largest cluster
        centroid = np.mean(largest_cluster_points, axis=0)

        # Reshape the centroid array to 2D
        centroid_2D = centroid.reshape(1, -1)

        ## randomly select a point from the largest cluster
        point_idx = np.random.choice(largest_cluster_points.shape[0], 1, replace=False)
        point = largest_cluster_points[point_idx]
        point_2D = point.reshape(1, -1)

        # Since the data was standardized, the centroid needs to be inverse_transformed to get back to the original data scale.
        centroid_original_scale = scaler.inverse_transform(centroid_2D)
        point_original_scale = scaler.inverse_transform(point_2D)

        return point_original_scale
    
    def get_smpl_trajectory_collision(self, all_poses3d_trajectory, all_initial_smpl_trajectory):
        """
        all_poses3d_trajectory is dict of human_name vs time x 17 x 4
        initial_smpl_trajectory is dict of human_name vs list[smpl_info] --> nested dicts. The list is time indexed
        """

        self.smplify_collision, self.smplify_collision_config = self.build_smplify_collision(human_names=all_poses3d_trajectory.keys())

        human_datas = {}

        ## convert all_poses3d_trajectory to the format of smplify
        for human_name in all_poses3d_trajectory.keys():
            poses3d_trajectory = all_poses3d_trajectory[human_name]
            total_time = poses3d_trajectory.shape[0]
            keypoints_src = poses3d_trajectory[:, :17, :3] ## take the first 17 keypoints, T x 17 x 3

            src_mask = np.ones(17)

            keypoints, mask =  convert_kps(
                                keypoints_src,
                                mask=src_mask,
                                src=self.keypoint_type,
                                dst=self.smplify_collision_config.body_model['keypoint_dst']
                            )
    
            keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)

            ##---------------as numpy array----------------------
            init_beta = [all_initial_smpl_trajectory[human_name][t-1]['betas'].reshape(1, -1) for t in range(1, total_time+1)] ## dim = T x 10
            init_body_pose = [all_initial_smpl_trajectory[human_name][t-1]['body_pose'].reshape(1, -1) for t in range(1, total_time+1)] ## dim = T x (69)
            init_transl = [all_initial_smpl_trajectory[human_name][t-1]['transl'].reshape(1, -1) for t in range(1, total_time+1)] ## dim = T x (3)
            init_global_orient = [all_initial_smpl_trajectory[human_name][t-1]['global_orient'].reshape(1, -1) for t in range(1, total_time+1)] ## dim = T x (3)
            
            init_body_pose = np.concatenate(init_body_pose, axis=0) ## T x 69
            init_transl = np.concatenate(init_transl, axis=0) ## T x 3
            init_global_orient = np.concatenate(init_global_orient, axis=0) ## T x 3
            init_beta = np.concatenate(init_beta, axis=0) ## T x 10

            ##---------------as torch tensor----------------------
            keypoints = torch.tensor(keypoints).float().to(self.device) ## dim = T x 45 x 3
            keypoints_conf = torch.tensor(keypoints_conf).float().to(self.device) ## dim = T x 45

            init_beta = torch.tensor(init_beta).float().to(self.device) ## dim = T x 1 x 10
            init_body_pose = torch.tensor(init_body_pose).float().to(self.device) ## dim = T x 69
            init_transl = torch.tensor(init_transl).float().to(self.device) ## dim = T x 3
            init_global_orient = torch.tensor(init_global_orient).float().to(self.device) ## dim = T x 3
            
            ##---------------smplify----------------------
            human_data = dict(
                            human_name=human_name,
                            keypoints3d=keypoints,
                            keypoints3d_conf=keypoints_conf,
                            init_betas=init_beta,
                            init_pose=init_body_pose,
                            init_body_pose=init_body_pose,
                            init_transl=init_transl,
                            init_global_orient=init_global_orient,
                        )
            
            human_datas[human_name] = human_data

        # run SMPLify(X)
        smplify_output = self.smplify_collision(human_datas, return_joints=True, return_verts=True)

        ret_dict = {}

        for human_name in all_poses3d_trajectory.keys():
            ret = {t:{} for t in range(total_time)}

            for key in smplify_output[human_name].keys():
                for t in range(total_time):
                    if key == 'epoch_loss':
                        ret[t][key] = smplify_output[human_name][key]
                    elif key == 'betas':
                        ret[t][key] = (smplify_output[human_name][key][0].cpu().numpy()).reshape(-1)
                    else:
                        ret[t][key] = smplify_output[human_name][key][t].cpu().numpy()
            
            ret_dict[human_name] = ret

        return ret_dict
