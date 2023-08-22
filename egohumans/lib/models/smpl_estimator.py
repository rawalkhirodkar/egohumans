import numpy as np
import os
import cv2
from tqdm import tqdm
import sys
import pathlib
import torch
import torchgeometry as tgm

##-------------------------------------------------------------------------
def add_path(path):  
    if path not in sys.path:
        sys.path.insert(0, path)
    return

cliff_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', '..', 'external', 'cliff') 
add_path(cliff_dir)

import smplx
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from common import constants
from common.utils import strip_prefix_if_present, cam_crop2full
from common.mocap_dataset import MocapDataset
from torch.utils.data import DataLoader

##------------------------------------------------------------------------------------
def get_smpl_faces():
    try:
        smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl")
        smpl_model_faces = smpl_model.faces

    except:
        print('environment warning! loading the smpl faces locally')
        assets_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', '..', 'assets') 
        smpl_model_faces = np.load(os.path.join(assets_dir, 'smpl_faces.npy'))

    return smpl_model_faces

##------------------------------------------------------------------------------------
class SMPLModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.cliff_model = self.load_model()

        # Setup the SMPL model
        self.smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(self.device)

        return 

    def load_model(self):
        cliff_model = cliff_hr48(constants.SMPL_MEAN_PARAMS).to(self.device)
        ckpt_path = os.path.join(cliff_dir, 'data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')
        print("Load the CLIFF checkpoint from path:", ckpt_path)
        state_dict = torch.load(ckpt_path)['model']
        state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        cliff_model.load_state_dict(state_dict, strict=True)
        cliff_model.eval()
        return cliff_model
    
    def get_initial_smpl(self, image_path, bboxes, bbox_padding=1.25):
        img_bgr_cliff = [cv2.imread(image_path)]
        bbox_cliff = np.zeros((len(bboxes), 8))

        for idx, human_name in enumerate(bboxes.keys()):
            bbox = bboxes[human_name]
            bbox_cliff[idx] = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3], 1, 1, 0]) # batch_id, min_x, min_y, max_x, max_y, det_conf, nms_conf, category_id

        mocap_db = MocapDataset(img_bgr_cliff, bbox_cliff)
        mocap_data_loader = DataLoader(mocap_db, batch_size=len(bboxes), num_workers=0)

        ## only one batch
        for batch in tqdm(mocap_data_loader):
            norm_img = batch["norm_img"].to(self.device).float()
            center = batch["center"].to(self.device).float()
            scale = batch["scale"].to(self.device).float()
            img_h = batch["img_h"].to(self.device).float()
            img_w = batch["img_w"].to(self.device).float()
            focal_length = batch["focal_length"].to(self.device).float()

            scale = scale*bbox_padding ## extend the bbox

            cx, cy, b = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
            bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_cam_crop = self.cliff_model(norm_img, bbox_info)

            # convert the camera parameters from the crop camera to the full camera
            full_img_shape = torch.stack((img_h, img_w), dim=-1)
            pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
            pred_vertices, pred_joints = self.get_vertices_and_joints(pred_betas, pred_rotmat, pred_cam_full)

            ## convert to axis angle format
            rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device).view(1, 3, 1)
            rot_pad = rot_pad.expand(pred_rotmat.shape[0] * 24, -1, -1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)  # N*72
            break

        smpl_params = {}
        pred_betas = pred_betas.detach().cpu().numpy()
        pred_rotmat = pred_rotmat.detach().cpu().numpy()
        pred_pose = pred_pose.detach().cpu().numpy()
        pred_cam_full = pred_cam_full.detach().cpu().numpy()
        focal_length = focal_length.detach().cpu().numpy()

        for idx, human_name in enumerate(bboxes.keys()):
            smpl_params[human_name] = {'betas': pred_betas[idx], \
                            'rotmat': pred_rotmat[idx], \
                            'pose': pred_pose[idx], \
                            'cam_full': pred_cam_full[idx], \
                            'vertices': pred_vertices[idx],
                            'joints': pred_joints[idx],
                            'focal_length': focal_length[idx],} ## beta: (10,), rotmat: (24, 3, 3)

        return smpl_params

    def get_vertices_and_joints(self, pred_betas, pred_rotmat, pred_cam_full):
        ####------info------
        ## pred_betas: N x 10
        ## pred_rotmat: N x 24 x 3 x 3
        ## pred_cam_full: N x 3
        ## pred_vertices: N x 6890 x 3
        # pred_output = self.smpl_model(betas=pred_betas,
        #                          body_pose=pred_rotmat[:, 1:],
        #                          global_orient=pred_rotmat[:, [0]],
        #                          pose2rot=False,
        #                          transl=pred_cam_full)
        # pred_vertices = pred_output.vertices.cpu().numpy()
        # pred_joints = pred_output.joints.cpu().numpy()

        ## throw away global information
        identity_global_orient = torch.Tensor(np.eye(3).reshape(1, 1, 3, 3)).to(pred_rotmat.device)
        identity_global_orient = identity_global_orient.repeat(len(pred_rotmat), 1, 1, 1)

        pred_output = self.smpl_model(betas=pred_betas,
                                 body_pose=pred_rotmat[:, 1:],
                                 global_orient=identity_global_orient,
                                 pose2rot=False,
                                 transl=None)
        pred_vertices = pred_output.vertices.detach().cpu().numpy()
        pred_joints = pred_output.joints.detach().cpu().numpy()

        return pred_vertices, pred_joints


    def get_initial_vertices(self, betas, body_pose_aa, global_orient_aa, transl):
        betas = torch.tensor(betas.reshape(1, -1), dtype=torch.float32, device=self.device)
        body_pose_aa = torch.tensor(body_pose_aa.reshape(1, -1), dtype=torch.float32, device=self.device)
        global_orient_aa = torch.tensor(global_orient_aa.reshape(1, -1), dtype=torch.float32, device=self.device)
        transl = torch.tensor(transl.reshape(1, -1), dtype=torch.float32, device=self.device)

        output = self.smpl_model(betas=betas,
                                 body_pose=body_pose_aa,
                                 global_orient=global_orient_aa,
                                 transl=transl)

        vertices = output.vertices[0].cpu().numpy()

        return vertices