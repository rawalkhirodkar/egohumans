from typing import List, Tuple, Union

import numpy as np
import torch
from mmcv.runner import build_optimizer

from mmhuman3d.core.cameras import build_cameras
from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
    get_keypoint_idxs_by_part,
)
from ..body_models.builder import build_body_model
from ..losses.builder import build_loss


class OptimizableParameters():
    """Collects parameters for optimization."""

    def __init__(self):
        self.opt_params = []

    def set_param(self, fit_param: torch.Tensor, param: torch.Tensor) -> None:
        """Set requires_grad and collect parameters for optimization.
        Args:
            fit_param: whether to optimize this body model parameter
            param: body model parameter
        Returns:
            None
        """
        if fit_param:
            param.requires_grad = True
            self.opt_params.append(param)
        else:
            param.requires_grad = False

    def parameters(self) -> List[torch.Tensor]:
        """Returns parameters. Compatible with mmcv's build_parameters()
        Returns:
            opt_params: a list of body model parameters for optimization
        """
        return self.opt_params


class SMPLifyCollision(object):
    """Re-implementation of SMPLify with extended features.
    - video input
    - 3D keypoints
    """

    def __init__(self,
                 body_model: Union[dict, torch.nn.Module],
                 num_epochs: int = 20,
                 camera: Union[dict, torch.nn.Module] = None,
                 img_res: Union[Tuple[int], int] = 224,
                 stages: dict = None,
                 optimizer: dict = None,
                 keypoints2d_loss: dict = None,
                 keypoints3d_loss: dict = None,
                 shape_prior_loss: dict = None,
                 joint_prior_loss: dict = None,
                 smooth_loss: dict = None,
                 pose_prior_loss: dict = None,
                 pose_reg_loss: dict = None,
                 crouch_reg_loss: dict = None,
                 collision_loss: dict = None,
                 limb_length_loss: dict = None,
                 use_one_betas_per_video: bool = False,
                 ignore_keypoints: List[int] = None,
                 device=torch.device(
                     'cuda' if torch.cuda.is_available() else 'cpu'),
                 verbose: bool = False) -> None:
        """
        Args:
            body_model: config or an object of body model.
            num_epochs: number of epochs of registration
            camera: config or an object of camera
            img_res: image resolution. If tuple, values are (width, height)
            stages: config of registration stages
            optimizer: config of optimizer
            keypoints2d_loss: config of keypoint 2D loss
            keypoints3d_loss: config of keypoint 3D loss
            shape_prior_loss: config of shape prior loss.
                Used to prevent extreme shapes.
            joint_prior_loss: config of joint prior loss.
                Used to prevent large joint rotations.
            smooth_loss: config of smooth loss.
                Used to prevent jittering by temporal smoothing.
            pose_prior_loss: config of pose prior loss.
                Used to prevent unnatural pose.
            pose_reg_loss: config of pose regularizer loss.
                Used to prevent pose being too large.
            limb_length_loss: config of limb length loss.
                Used to prevent the change of body shape.
            use_one_betas_per_video: whether to use the same beta parameters
                for all frames in a single video sequence.
            ignore_keypoints: list of keypoint names to ignore in keypoint
                loss computation
            device: torch device
            verbose: whether to print information during registration
        Returns:
            None
        """
        self.use_one_betas_per_video = use_one_betas_per_video
        self.num_epochs = num_epochs
        self.img_res = img_res
        self.device = device
        self.stage_config = stages
        self.optimizer = optimizer
        self.keypoints2d_mse_loss = build_loss(keypoints2d_loss)
        self.keypoints3d_mse_loss = build_loss(keypoints3d_loss)
        self.shape_prior_loss = build_loss(shape_prior_loss)
        self.joint_prior_loss = build_loss(joint_prior_loss)
        self.smooth_loss = build_loss(smooth_loss)
        self.pose_prior_loss = build_loss(pose_prior_loss)
        self.pose_reg_loss = build_loss(pose_reg_loss)
        self.crouch_reg_loss = build_loss(crouch_reg_loss)
        self.limb_length_loss = build_loss(limb_length_loss)
        self.collision_loss = build_loss(collision_loss)

        if self.joint_prior_loss is not None:
            self.joint_prior_loss = self.joint_prior_loss.to(self.device)
        if self.smooth_loss is not None:
            self.smooth_loss = self.smooth_loss.to(self.device)
        if self.pose_prior_loss is not None:
            self.pose_prior_loss = self.pose_prior_loss.to(self.device)
        if self.pose_reg_loss is not None:
            self.pose_reg_loss = self.pose_reg_loss.to(self.device)
        if self.crouch_reg_loss is not None:
            self.crouch_reg_loss = self.crouch_reg_loss.to(self.device)
        if self.limb_length_loss is not None:
            self.limb_length_loss = self.limb_length_loss.to(self.device)
        if self.collision_loss is not None:
            self.collision_loss = self.collision_loss.to(self.device)

        # initialize body model
        if isinstance(body_model, dict):
            self.body_model = build_body_model(body_model).to(self.device)
        elif isinstance(body_model, torch.nn.Module):
            self.body_model = body_model.to(self.device)
        else:
            raise TypeError(f'body_model should be either dict or '
                            f'torch.nn.Module, but got {type(body_model)}')

        # initialize camera
        if camera is not None:
            if isinstance(camera, dict):
                self.camera = build_cameras(camera).to(self.device)
            elif isinstance(camera, torch.nn.Module):
                self.camera = camera.to(device)
            else:
                raise TypeError(f'camera should be either dict or '
                                f'torch.nn.Module, but got {type(camera)}')

        self.ignore_keypoints = ignore_keypoints
        self.verbose = verbose
        self.msg = ''

        self._set_keypoint_idxs()

    def __call__(self,
                 human_datas,
                 return_verts: bool = True,
                 return_joints: bool = True,
                 return_full_pose: bool = False,
                 return_losses: bool = False,
                 ) -> dict:
        """Run registration.
        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
            Provide only keypoints2d or keypoints3d, not both.
        Args:
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            init_global_orient: initial global_orient of shape (B, 3)
            init_transl: initial transl of shape (B, 3)
            init_body_pose: initial body_pose of shape (B, 69)
            init_betas: initial betas of shape (B, D)
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            return_losses: whether to return loss dict
        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """

        global_orient_dict = {}
        transl_dict = {}
        body_pose_dict = {}
        betas_dict = {}
        keypoints3d_dict = {}
        keypoints3d_conf_dict = {}

        for human_name in human_datas.keys():
            human_data = human_datas[human_name]

            keypoints2d = None
            keypoints2d_conf = None
            keypoints3d = human_data['keypoints3d']
            keypoints3d_conf = human_data['keypoints3d_conf']
            init_global_orient = human_data['init_global_orient']
            init_transl = human_data['init_transl']
            init_body_pose = human_data['init_body_pose']
            init_betas = human_data['init_betas']

            assert keypoints2d is not None or keypoints3d is not None, \
                'Neither of 2D nor 3D keypoints are provided.'
            assert not (keypoints2d is not None and keypoints3d is not None), \
                'Do not provide both 2D and 3D keypoints.'
            batch_size = keypoints2d.shape[0] if keypoints2d is not None \
                else keypoints3d.shape[0]

            global_orient = self._match_init_batch_size(init_global_orient, self.body_model.global_orient, batch_size)
            transl = self._match_init_batch_size(init_transl, self.body_model.transl, batch_size)
            body_pose = self._match_init_batch_size(init_body_pose, self.body_model.body_pose, batch_size)

            if init_betas is None and self.use_one_betas_per_video:
                betas = torch.zeros(1, self.body_model.betas.shape[-1]).to(
                    self.device)
            
            elif init_betas is not None and self.use_one_betas_per_video:
                betas = init_betas[0].reshape(1, -1).to(self.device)
            else:
                betas = self._match_init_batch_size(init_betas, self.body_model.betas, batch_size)
            
            ## add to dict
            global_orient_dict[human_name] = global_orient
            transl_dict[human_name] = transl
            body_pose_dict[human_name] = body_pose
            betas_dict[human_name] = betas
            keypoints3d_dict[human_name] = keypoints3d
            keypoints3d_conf_dict[human_name] = keypoints3d_conf

        ##----------------- start optimization -----------------##        
        last_epoch_loss = 0

        for i in range(self.num_epochs):
            epoch_loss = 0
            for stage_idx, stage_config in enumerate(self.stage_config):
                if self.verbose:
                    
                    ## check if self.msg_template_dict attribute exists
                    if not hasattr(self, 'msg_template_dict'):
                        self.msg_template_dict = {}

                    for human_name in human_datas.keys():
                        self.msg_template_dict[human_name] = '{}, epoch {}, stage {}, iter dummy:'.format(human_name, i, stage_idx)

                loss = self._optimize_stage(
                    global_orient_dict=global_orient_dict,
                    transl_dict=transl_dict,
                    body_pose_dict=body_pose_dict,
                    betas_dict=betas_dict,
                    keypoints3d_dict=keypoints3d_dict,
                    keypoints3d_conf_dict=keypoints3d_conf_dict,
                    **stage_config,
                )

                if loss is not None:
                    epoch_loss += loss

            last_epoch_loss = epoch_loss

        # collate results
        ret_dict = {}

        for human_name in human_datas.keys():
            global_orient = global_orient_dict[human_name]
            transl = transl_dict[human_name]
            body_pose = body_pose_dict[human_name]
            betas = betas_dict[human_name]

            ret = {
                'global_orient': global_orient,
                'transl': transl,
                'body_pose': body_pose,
                'betas': betas, 
            }

            ret_dict[human_name] = ret

        if return_verts or return_joints or \
                return_full_pose or return_losses:
            eval_ret = self.evaluate(
                global_orient_dict=global_orient_dict,
                body_pose_dict=body_pose_dict,
                betas_dict=betas_dict,
                transl_dict=transl_dict,
                keypoints3d_dict=keypoints3d_dict,
                keypoints3d_conf_dict=keypoints3d_conf_dict,
                return_verts=return_verts,
                return_full_pose=return_full_pose,
                return_joints=return_joints,
                reduction_override='none'  # sample-wise loss
            )

            if return_verts:
                for human_name in human_datas.keys():
                    ret_dict[human_name]['vertices'] = eval_ret[human_name]['vertices']

            if return_joints:
                for human_name in human_datas.keys():
                    ret_dict[human_name]['joints'] = eval_ret[human_name]['joints']

            if return_full_pose:
                for human_name in human_datas.keys():
                    ret_dict[human_name]['full_pose'] = eval_ret[human_name]['full_pose']

            if return_losses:
                for human_name in human_datas.keys():
                    for k in eval_ret[human_name].keys():
                        if 'loss' in k:
                            ret_dict[human_name][k] = eval_ret[human_name][k]

        ## detach and clone
        for human_name in human_datas.keys():
            for k, v in ret_dict[human_name].items():
                if isinstance(v, torch.Tensor):
                    ret_dict[human_name][k] = v.detach().clone()

        return ret_dict

    def _optimize_stage(self,
                        betas_dict: dict,
                        body_pose_dict: dict,
                        global_orient_dict: dict,
                        transl_dict: dict,
                        fit_global_orient: bool = True,
                        fit_transl: bool = True,
                        fit_body_pose: bool = True,
                        fit_betas: bool = True,
                        keypoints3d_dict: dict = None,
                        keypoints3d_conf_dict: dict = None,
                        keypoints3d_weight: float = None,
                        shape_prior_weight: float = None,
                        joint_prior_weight: float = None,
                        smooth_loss_weight: float = None,
                        pose_prior_weight: float = None,
                        pose_reg_weight: float = None,
                        crouch_reg_weight: float = None,
                        collision_loss_weight: float = None,
                        limb_length_weight: float = None,
                        joint_weights: dict = {},
                        num_iter: int = 1,
                        ftol: float = 1e-4,
                        **kwargs) -> None:
        """Optimize a stage of body model parameters according to
        configuration.
        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            fit_global_orient: whether to optimize global_orient
            fit_transl: whether to optimize transl
            fit_body_pose: whether to optimize body_pose
            fit_betas: whether to optimize betas
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            pose_reg_weight: weight of pose regularization loss
            limb_length_weight: weight of limb length loss
            joint_weights: per joint weight of shape (K, )
            num_iter: number of iterations
            ftol: early stop tolerance for relative change in loss
        Returns:
            None
        """
        parameters = OptimizableParameters()

        ## set parametrs for each human

        for human_name in global_orient_dict.keys():
            global_orient = global_orient_dict[human_name]
            transl = transl_dict[human_name]
            body_pose = body_pose_dict[human_name]
            betas = betas_dict[human_name]

            parameters.set_param(fit_global_orient, global_orient)
            parameters.set_param(fit_transl, transl)
            parameters.set_param(fit_body_pose, body_pose)
            parameters.set_param(fit_betas, betas)

        optimizer = build_optimizer(parameters, self.optimizer)

        ##------------------   
        pre_loss = None

        for iter_idx in range(num_iter):

            for human_name in global_orient_dict.keys():

                ## check if self.msg_dict attribute exists
                if not hasattr(self, 'msg_dict'):
                    self.msg_dict = {}
                
                self.msg_dict[human_name] = self.msg_template_dict[human_name].replace('dummy', '{}/{}'.format(iter_idx, num_iter))

            def closure():
                optimizer.zero_grad()

                betas_video_dict = {}
                for human_name in betas_dict.keys():
                    betas_video_dict[human_name] = self._expand_betas(body_pose_dict[human_name].shape[0], betas_dict[human_name])

                loss_dict = self.evaluate(
                    global_orient_dict=global_orient_dict,
                    body_pose_dict=body_pose_dict,
                    betas_dict=betas_video_dict,
                    transl_dict=transl_dict,
                    keypoints3d_dict=keypoints3d_dict,
                    keypoints3d_conf_dict=keypoints3d_conf_dict,
                    keypoints3d_weight=keypoints3d_weight,
                    joint_prior_weight=joint_prior_weight,
                    shape_prior_weight=shape_prior_weight,
                    smooth_loss_weight=smooth_loss_weight,
                    pose_prior_weight=pose_prior_weight,
                    pose_reg_weight=pose_reg_weight,
                    crouch_reg_weight=crouch_reg_weight,
                    collision_loss_weight=collision_loss_weight,
                    limb_length_weight=limb_length_weight,
                    joint_weights=joint_weights,
                    return_verts=True,
                )
                
                loss = 0
                ## sum over all humans
                for human_name in loss_dict.keys():
                    loss += loss_dict[human_name]['total_loss']
                
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            if iter_idx > 0 and pre_loss is not None and ftol > 0:
                loss_rel_change = self._compute_relative_change(
                    pre_loss, loss.item())
                if loss_rel_change < ftol:
                    if self.verbose:
                        print(f'[ftol={ftol}] Early stop at {iter_idx} iter!')
                    break
            pre_loss = loss.item()

        return pre_loss

    def evaluate(
        self,
        betas_dict: dict = {},
        body_pose_dict: dict = {},
        global_orient_dict: dict = {},
        transl_dict: dict = {},
        keypoints3d_dict: dict = {},
        keypoints3d_conf_dict: dict = {},
        keypoints3d_weight: float = None,
        shape_prior_weight: float = None,
        joint_prior_weight: float = None,
        smooth_loss_weight: float = None,
        pose_prior_weight: float = None,
        pose_reg_weight: float = None,
        crouch_reg_weight: float = None,
        collision_loss_weight: float = None,
        limb_length_weight: float = None,
        joint_weights: dict = {},
        return_verts: bool = False,
        return_full_pose: bool = False,
        return_joints: bool = False,
        reduction_override: str = None,
    ) -> dict:
        """Evaluate fitted parameters through loss computation. This function
        serves two purposes: 1) internally, for loss backpropagation 2)
        externally, for fitting quality evaluation.
        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
        Args:
            betas: shape (B, D)
            body_pose: shape (B, 69)
            global_orient: shape (B, 3)
            transl: shape (B, 3)
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            pose_reg_weight: weight of pose regularization loss
            limb_length_weight: weight of limb length loss
            joint_weights: per joint weight of shape (K, )
            return_verts: whether to return vertices
            return_joints: whether to return joints
            return_full_pose: whether to return full pose
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'
        Returns:
            ret: a dictionary that includes body model parameters,
                and optional attributes such as vertices and joints
        """

        ret_dict = {}

        body_model_output_dict = {}
        model_joints_dict = {}
        model_joint_mask_dict = {}
        vertices_dict = {}

        assert return_verts is True

        for human_name in betas_dict.keys():
            body_model_output_dict[human_name] = self.body_model(
                global_orient=global_orient_dict[human_name],
                body_pose=body_pose_dict[human_name],
                betas=betas_dict[human_name],
                transl=transl_dict[human_name],
                return_verts=return_verts,
                return_full_pose=return_full_pose)

            model_joints_dict[human_name] = body_model_output_dict[human_name]['joints']
            model_joint_mask_dict[human_name] = body_model_output_dict[human_name]['joint_mask']
            vertices_dict[human_name] = body_model_output_dict[human_name]['vertices']

        all_loss_dict = self._compute_loss(
            model_joints_dict=model_joints_dict,
            model_joint_conf_dict=model_joint_mask_dict,
            vertices_dict=vertices_dict,
            keypoints3d_dict=keypoints3d_dict,
            keypoints3d_conf_dict=keypoints3d_conf_dict,
            keypoints3d_weight=keypoints3d_weight,
            joint_prior_weight=joint_prior_weight,
            shape_prior_weight=shape_prior_weight,
            smooth_loss_weight=smooth_loss_weight,
            pose_prior_weight=pose_prior_weight,
            pose_reg_weight=pose_reg_weight,
            crouch_reg_weight=crouch_reg_weight,
            collision_loss_weight=collision_loss_weight,
            limb_length_weight=limb_length_weight,
            joint_weights=joint_weights,
            reduction_override=reduction_override,
            global_orient_dict=global_orient_dict,
            body_pose_dict=body_pose_dict,
            betas_dict=betas_dict,)

        for human_name in all_loss_dict.keys():
            ret = {}
            ret.update(all_loss_dict[human_name])

            if return_verts:
                ret['vertices'] = body_model_output_dict[human_name]['vertices']
            if return_full_pose:
                ret['full_pose'] = body_model_output_dict[human_name]['full_pose']
            if return_joints:
                ret['joints'] = model_joints_dict[human_name]
            
            ret_dict[human_name] = ret

        return ret_dict

    def _compute_loss(self,
                      model_joints_dict: dict = {},
                      model_joint_conf_dict: dict = {},
                      vertices_dict: dict = {},
                      keypoints3d_dict: dict = {},
                      keypoints3d_conf_dict: dict = {},
                      keypoints3d_weight: float = None,
                      shape_prior_weight: float = None,
                      joint_prior_weight: float = None,
                      smooth_loss_weight: float = None,
                      pose_prior_weight: float = None,
                      pose_reg_weight: float = None,
                      crouch_reg_weight: float = None,
                      collision_loss_weight: float = None,
                      limb_length_weight: float = None,
                      joint_weights: dict = {},
                      reduction_override: str = None,
                      global_orient_dict: dict = {},
                      body_pose_dict: dict = {},
                      betas_dict: dict = {},):
        """Loss computation.
        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
        Args:
            model_joints: 3D joints regressed from body model of shape (B, K)
            model_joint_conf: 3D joint confidence of shape (B, K). It is
                normally all 1, except for zero-pads due to convert_kps in
                the SMPL wrapper.
            keypoints2d: 2D keypoints of shape (B, K, 2)
            keypoints2d_conf: 2D keypoint confidence of shape (B, K)
            keypoints2d_weight: weight of 2D keypoint loss
            keypoints3d: 3D keypoints of shape (B, K, 3).
            keypoints3d_conf: 3D keypoint confidence of shape (B, K)
            keypoints3d_weight: weight of 3D keypoint loss
            shape_prior_weight: weight of shape prior loss
            joint_prior_weight: weight of joint prior loss
            smooth_loss_weight: weight of smooth loss
            pose_prior_weight: weight of pose prior loss
            joint_weights: per joint weight of shape (K, )
            reduction_override: reduction method, e.g., 'none', 'sum', 'mean'
            body_pose: shape (B, 69), for loss computation
            betas: shape (B, D), for loss computation
        Returns:
            losses: a dict that contains all losses
        """
        losses_dict = {}

        ## compute the collision loss separately as it is multi-human. Rest of the losses are single human
        ##-----------------------------------------------------------------------------------------------------------------
        ## prepare the faces for collision loss

        mesh_faces = self.body_model.faces.copy() ## (13776, 3), numpy array
        mesh_faces = torch.tensor(mesh_faces.astype(np.int32)).to(self.device)

        if not self._skip_loss(self.collision_loss, collision_loss_weight):
            collision_loss = self.collision_loss(
                vertices_dict=vertices_dict,
                mesh_faces=mesh_faces,
                loss_weight_override=collision_loss_weight,
                reduction_override=reduction_override)
            
        for human_name in model_joints_dict.keys():
            losses_dict[human_name] = {}
            losses_dict[human_name]['collision_loss'] = collision_loss[human_name]

        ##-----------------------------------------------------------------------------------------------------------------
        for human_name in model_joints_dict.keys():

            model_joints = model_joints_dict[human_name]
            model_joint_conf = model_joint_conf_dict[human_name]
            keypoints3d = keypoints3d_dict[human_name]
            keypoints3d_conf = keypoints3d_conf_dict[human_name]
            global_orient = global_orient_dict[human_name]
            body_pose = body_pose_dict[human_name]
            betas = betas_dict[human_name]

            losses = {}

            weight = self._get_weight(**joint_weights)

            ## the actual weighted loss is multiplication of pred_conf and target_conf and keypoint_weight
            # 3D keypoint loss
            if keypoints3d is not None and not self._skip_loss(
                    self.keypoints3d_mse_loss, keypoints3d_weight):
                keypoints3d_loss = self.keypoints3d_mse_loss(
                    pred=model_joints,
                    pred_conf=model_joint_conf,
                    target=keypoints3d,
                    target_conf=keypoints3d_conf,
                    keypoint_weight=weight,
                    loss_weight_override=keypoints3d_weight,
                    reduction_override=reduction_override)
                losses['keypoints3d_loss'] = keypoints3d_loss

            # regularizer to prevent betas from taking large values
            if not self._skip_loss(self.shape_prior_loss, shape_prior_weight):
                shape_prior_loss = self.shape_prior_loss(
                    betas=betas,
                    loss_weight_override=shape_prior_weight,
                    reduction_override=reduction_override)
                losses['shape_prior_loss'] = shape_prior_loss

            # joint prior loss
            if not self._skip_loss(self.joint_prior_loss, joint_prior_weight):
                joint_prior_loss = self.joint_prior_loss(
                    body_pose=body_pose,
                    loss_weight_override=joint_prior_weight,
                    reduction_override=reduction_override)
                losses['joint_prior_loss'] = joint_prior_loss

            # smooth body loss
            if not self._skip_loss(self.smooth_loss, smooth_loss_weight):
                smooth_loss = self.smooth_loss(
                    body_pose=body_pose,
                    loss_weight_override=smooth_loss_weight,
                    reduction_override=reduction_override)
                losses['smooth_loss'] = smooth_loss

            # pose prior loss
            if not self._skip_loss(self.pose_prior_loss, pose_prior_weight):
                pose_prior_loss = self.pose_prior_loss(
                    body_pose=body_pose,
                    loss_weight_override=pose_prior_weight,
                    reduction_override=reduction_override)
                losses['pose_prior_loss'] = pose_prior_loss

            # pose reg loss
            if not self._skip_loss(self.pose_reg_loss, pose_reg_weight):
                pose_reg_loss = self.pose_reg_loss(
                    body_pose=body_pose,
                    loss_weight_override=pose_reg_weight,
                    reduction_override=reduction_override)
                losses['pose_reg_loss'] = pose_reg_loss

            # crouch reg loss
            if not self._skip_loss(self.crouch_reg_loss, crouch_reg_weight):
                crouch_reg_loss = self.crouch_reg_loss(
                    body_pose=body_pose,
                    loss_weight_override=crouch_reg_weight,
                    reduction_override=reduction_override)
                losses['crouch_reg_loss'] = crouch_reg_loss

            # limb length loss
            if not self._skip_loss(self.limb_length_loss, limb_length_weight):
                limb_length_loss = self.limb_length_loss(
                    pred=model_joints,
                    pred_conf=model_joint_conf,
                    target=keypoints3d,
                    target_conf=keypoints3d_conf,
                    loss_weight_override=limb_length_weight,
                    reduction_override=reduction_override)
                losses['limb_length_loss'] = limb_length_loss

            losses_dict[human_name].update(losses)

        for human_name in model_joints_dict.keys():
            losses = losses_dict[human_name]

            total_loss = 0

            for loss_name, loss in losses.items():
                if loss.ndim == 3:
                    total_loss = total_loss + loss.sum(dim=(2, 1))
                elif loss.ndim == 2:
                    total_loss = total_loss + loss.sum(dim=-1)
                else:
                    total_loss = total_loss + loss

            losses_dict[human_name]['total_loss'] = total_loss

            if self.verbose:
                msg = self.msg_dict[human_name] + ' '
                for loss_name, loss in losses_dict[human_name].items():
                    msg += f'{loss_name}={loss.mean().item():.6f}, '
                if self.verbose:
                    print(msg.strip(', '))

        return losses_dict

    def _match_init_batch_size(self, init_param: torch.Tensor,
                               init_param_body_model: torch.Tensor,
                               batch_size: int) -> torch.Tensor:
        """A helper function to ensure body model parameters have the same
        batch size as the input keypoints.
        Args:
            init_param: input initial body model parameters, may be None
            init_param_body_model: initial body model parameters from the
                body model
            batch_size: batch size of keypoints
        Returns:
            param: body model parameters with batch size aligned
        """

        # param takes init values
        param = init_param.detach().clone() \
            if init_param is not None \
            else init_param_body_model.detach().clone()

        # expand batch dimension to match batch size
        param_batch_size = param.shape[0]
        if param_batch_size != batch_size:
            if param_batch_size == 1:
                param = param.repeat(batch_size, *[1] * (param.ndim - 1))
            else:
                raise ValueError('Init param does not match the batch size of '
                                 'keypoints, and is not 1.')

        # shape check
        assert param.shape[0] == batch_size
        assert param.shape[1:] == init_param_body_model.shape[1:], \
            f'Shape mismatch: {param.shape} vs {init_param_body_model.shape}'

        return param

    def _set_keypoint_idxs(self) -> None:
        """Set keypoint indices to 1) body parts to be assigned different
        weights 2) be ignored for keypoint loss computation.
        Returns:
            None
        """
        convention = self.body_model.keypoint_dst

        # obtain ignore keypoint indices
        if self.ignore_keypoints is not None:
            self.ignore_keypoint_idxs = []
            for keypoint_name in self.ignore_keypoints:
                keypoint_idx = get_keypoint_idx(
                    keypoint_name, convention=convention)
                if keypoint_idx != -1:
                    self.ignore_keypoint_idxs.append(keypoint_idx)

        # obtain body part keypoint indices
        shoulder_keypoint_idxs = get_keypoint_idxs_by_part(
            'shoulder', convention=convention)
        hip_keypoint_idxs = get_keypoint_idxs_by_part(
            'hip', convention=convention)
        self.shoulder_hip_keypoint_idxs = [
            *shoulder_keypoint_idxs, *hip_keypoint_idxs
        ]

    def _get_weight(self,
                    use_shoulder_hip_only: bool = False,
                    body_weight: float = 1.0) -> torch.Tensor:
        """Get per keypoint weight.
        Notes:
            K: number of keypoints
        Args:
            use_shoulder_hip_only: whether to use only shoulder and hip
                keypoints for loss computation. This is useful in the
                warming-up stage to find a reasonably good initialization.
            body_weight: weight of body keypoints. Body part segmentation
                definition is included in the HumanData convention.
        Returns:
            weight: per keypoint weight tensor of shape (K)
        """

        num_keypoint = self.body_model.num_joints

        if use_shoulder_hip_only:
            weight = torch.zeros([num_keypoint]).to(self.device)
            weight[self.shoulder_hip_keypoint_idxs] = 1.0
            weight = weight * body_weight
        else:
            weight = torch.ones([num_keypoint]).to(self.device)
            weight = weight * body_weight

        if hasattr(self, 'ignore_keypoint_idxs'):
            weight[self.ignore_keypoint_idxs] = 0.0

        return weight

    def _expand_betas(self, batch_size, betas):
        """A helper function to expand the betas's first dim to match batch
        size such that the same beta parameters can be used for all frames in a
        video sequence.
        Notes:
            B: batch size
            K: number of keypoints
            D: shape dimension
        Args:
            batch_size: batch size
            betas: shape (B, D)
        Returns:
            betas_video: expanded betas
        """
        # no expansion needed
        if batch_size == betas.shape[0]:
            return betas

        # first dim is 1
        else:
            feat_dim = betas.shape[-1]
            betas_video = betas.view(1, feat_dim).expand(batch_size, feat_dim)

        return betas_video

    @staticmethod
    def _compute_relative_change(pre_v, cur_v):
        """Compute relative loss change. If relative change is small enough, we
        can apply early stop to accelerate the optimization. (1) When one of
        the value is larger than 1, we calculate the relative change by diving
        their max value. (2) When both values are smaller than 1, it degrades
        to absolute change. Intuitively, if two values are small and close,
        dividing the difference by the max value may yield a large value.
        Args:
            pre_v: previous value
            cur_v: current value
        Returns:
            float: relative change
        """
        return np.abs(pre_v - cur_v) / max([np.abs(pre_v), np.abs(cur_v), 1])

    @staticmethod
    def _skip_loss(loss, loss_weight_override):
        """Whether to skip loss computation. If loss is None, it will directly
        skip the loss to avoid RuntimeError. If loss is not None, the table
        below shows the return value. If the return value is True, it means the
        computation of loss can be skipped. As the result is 0 even if it is
        calculated, we can skip it to save computational cost.
        | loss.loss_weight | loss_weight_override | returns |
        | ---------------- | -------------------- | ------- |
        |      == 0        |         None         |   True  |
        |      != 0        |         None         |   False |
        |      == 0        |         == 0         |   True  |
        |      != 0        |         == 0         |   True  |
        |      == 0        |         != 0         |   False |
        |      != 0        |         != 0         |   False |
        Args:
            loss: loss is an object that has attribute loss_weight.
                loss.loss_weight is assigned when loss is initialized.
            loss_weight_override: loss_weight used to override loss.loss_weight
        Returns:
            bool: True means skipping loss computation, and vice versa
        """
        if (loss is None) or (loss.loss_weight == 0 and loss_weight_override is
                              None) or (loss_weight_override == 0):
            return True
        return False