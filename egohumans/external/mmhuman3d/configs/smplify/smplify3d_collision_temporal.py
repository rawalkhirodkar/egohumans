type = 'SMPLifyCollision'
verbose = True

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
        num_iter=10,
        ftol=1e-4,
        fit_global_orient=False,
        fit_transl=False,
        fit_body_pose=True, 
        fit_betas=False,
        keypoints3d_weight=10.0,
        crouch_reg_weight=0.0,
        pose_reg_weight=0.001,
        smooth_loss_weight=1.0,
        limb_length_weight=1.0,
        shape_prior_weight=0.0,
        collision_loss_weight=1,
        joint_weights=dict(body_weight=1.0, use_shoulder_hip_only=False)
    ),

]

# optimizer = dict(type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe')
optimizer = dict(type='LBFGS', max_iter=20, lr=1e-2, line_search_fn='strong_wolfe') ## very aggressive learning rate, max iters for the optimizer

keypoints3d_loss = dict(
    type='KeypointMSELoss', loss_weight=10, reduction='sum', sigma=100)

shape_prior_loss = dict(
    type='ShapePriorLoss', loss_weight=5e-3, reduction='mean')

limb_length_loss = dict(
    type='LimbLengthLoss', convention='smpl', loss_weight=1., reduction='mean')

pose_reg_loss = dict(type='PoseRegLoss', loss_weight=0.001, reduction='mean')

crouch_reg_loss = dict(type='CrouchRegLoss', loss_weight=1.0, reduction='mean')

collision_loss = dict(type='CollisionLoss', grid_size=32, scale_factor=0.2, loss_weight=1.0, reduction='mean')

smooth_loss = dict(
    type='SmoothJointLoss', loss_weight=1.0, reduction='mean', loss_func='L2')

ignore_keypoints = [
    'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
]