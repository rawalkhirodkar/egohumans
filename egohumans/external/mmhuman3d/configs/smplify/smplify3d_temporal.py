type = 'SMPLify'
verbose = True
# verbose = False

body_model = dict(
    type='SMPL',
    gender='neutral',
    num_betas=10,
    keypoint_src='smpl_45',
    keypoint_dst='smpl_45',
    model_path='data/body_models/smpl',
    batch_size=1)

stages = [
    dict(
        num_iter=50,
        ftol=1e-4,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=False,
        fit_betas=False,
        keypoints3d_weight=1.0,
        pose_reg_weight=0.0,
        crouch_reg_weight=0.0,
        smooth_loss_weight=0.0,
        limb_length_weight=0.0,
        shape_prior_weight=0.0,
        joint_weights=dict(
            body_weight=5.0,
            use_shoulder_hip_only=True,
        )),
    ###stage 1: optimize `betas`,
    dict(
        num_iter=20, 
        ftol=1e-4,
        fit_global_orient=False,
        fit_transl=False,
        fit_body_pose=False,
        fit_betas=True,
        keypoints3d_weight=0.0,
        pose_reg_weight=0.0,
        crouch_reg_weight=0.0,
        smooth_loss_weight=0.0,
        limb_length_weight=1.0, ## fits the shape to the target limb length using keypoints
        shape_prior_weight=5e-3,
    ),

    dict(
        num_iter=120,
        ftol=1e-4,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True, 
        fit_betas=False,
        keypoints3d_weight=10.0,
        crouch_reg_weight=0.0,
        pose_reg_weight=0.001,
        smooth_loss_weight=1.0,
        limb_length_weight=1.0,
        shape_prior_weight=0.0,
        joint_weights=dict(body_weight=1.0, use_shoulder_hip_only=False)
    ),
]


optimizer = dict(
    type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe')

keypoints3d_loss = dict(
    type='KeypointMSELoss', loss_weight=10, reduction='sum', sigma=100)

shape_prior_loss = dict(
    type='ShapePriorLoss', loss_weight=5e-3, reduction='mean')

limb_length_loss = dict(
    type='LimbLengthLoss', convention='smpl', loss_weight=1., reduction='mean')

pose_reg_loss = dict(type='PoseRegLoss', loss_weight=0.001, reduction='mean')

crouch_reg_loss = dict(type='CrouchRegLoss', loss_weight=1.0, reduction='mean')

smooth_loss = dict(
    type='SmoothJointLoss', loss_weight=1.0, reduction='mean', loss_func='L2')

ignore_keypoints = [
    'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
]