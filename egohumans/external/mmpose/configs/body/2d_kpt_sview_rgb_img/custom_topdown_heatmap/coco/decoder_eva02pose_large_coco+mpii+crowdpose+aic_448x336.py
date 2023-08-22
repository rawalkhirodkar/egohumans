_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/coco.py',
]

# evaluate_every_n_epochs = 10 ## default
# evaluate_every_n_epochs = 2 
evaluate_every_n_epochs = 1

data_image_resolution = (448, 336) ## height, width. 4:3 
model_image_resolution = (448, 336) ## model architecture 

heatmap_resolution = (128, 96) ## ratio of 3.5 compared to image_resolution

depth = 24
embed_dim = 1024
decoder_embed_dim = 256

learning_rate = 1e-3 ## used for vit-h+

total_epochs = 100
decay_step_epochs = [40, 80]

sigma = 3 ## default

vis_every_iters = 500 ## default
# vis_every_iters = 20 ## debug

# use_gt_bbox = True
use_gt_bbox = False

# bbox_file = 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json' # #default
bbox_file = 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_70_person.json'

evaluation = dict(interval=evaluate_every_n_epochs, metric='mAP', save_best='AP')
checkpoint_config = dict(interval=evaluate_every_n_epochs, max_keep_ckpts=20) ## only keep last 10 checkpoints

## make sure the num_layers here is equal to the depth in the model config
optimizer = dict(type='AdamW', lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.1,
                 constructor='DecoderLayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                                    num_layers=depth, 
                                    layer_decay_rate=0.8,
                                    custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            },
                                    num_decoder_layers=6,
                                    decoder_layer_decay_rate=0.8,
                                    decoder_custom_keys={
                                            'bias': dict(decay_multi=0.),
                                            'pos_embed': dict(decay_mult=0.),
                                            'relative_position_bias_table': dict(decay_mult=0.),
                                            'norm': dict(decay_mult=0.)
                                            }
                                     ),
                )

optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=decay_step_epochs)
total_epochs = total_epochs
target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='Eva',
        img_size=model_image_resolution,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=16,
        use_eva02=True,
        qkv_bias=True,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=False,
        mlp_ratio=2.6666666666666665,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapDecoderHead',
        encoder_embed_dim=embed_dim,
        decoder_embed_dim=decoder_embed_dim,
        num_deconv_layers=2,
        num_deconv_filters=(decoder_embed_dim, decoder_embed_dim),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1, ),
        num_joints=channel_cfg['num_output_channels'],
        num_layers=6,
        num_heads=8,
        dropout=0.0,
        num_patch_h=model_image_resolution[0] // 14,
        num_patch_w=model_image_resolution[1] // 14,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

data_cfg = dict(
    image_size=[data_image_resolution[1], data_image_resolution[0]],
    heatmap_size=[heatmap_resolution[1], heatmap_resolution[0]],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=use_gt_bbox,
    det_bbox_thr=0.0,
    bbox_file=bbox_file,
)

mpii_data_cfg = dict(
    image_size=[data_image_resolution[1], data_image_resolution[0]],
    heatmap_size=[heatmap_resolution[1], heatmap_resolution[0]],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    dataset_idx=2,
    use_gt_bbox=True,
    bbox_file=None,
)

aic_data_cfg = dict(
    image_size=[data_image_resolution[1], data_image_resolution[0]],
    heatmap_size=[heatmap_resolution[1], heatmap_resolution[0]],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=None,
)

crowdpose_data_cfg = dict(
    image_size=[data_image_resolution[1], data_image_resolution[0]],
    heatmap_size=[heatmap_resolution[1], heatmap_resolution[0]],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    dataset_idx=3,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=None,
)

mpii_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=sigma,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'flip_pairs', 'dataset_idx'
        ]),
]

crowdpose_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=6,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=sigma, ## sigma for the gaussian heatmap, 2 for 256x192, 3 for 384x288 and higher
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

aic_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=sigma, ## sigma for the gaussian heatmap, 2 for 256x192, 3 for 384x288 and higher
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=sigma, ## sigma for the gaussian heatmap, 2 for 256x192, 3 for 384x288 and higher
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'
mpii_data_root = 'data/mpii'
crowdpose_data_root = 'data/crowdpose'
aic_data_root = 'data/aic'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=[
        dict(
            type='TopDownCocoDataset',
            ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
            img_prefix=f'{data_root}/train2017/',
            data_cfg=data_cfg,
            pipeline=train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='TopDownMpii2CocoDataset',
            ann_file=f'{mpii_data_root}/annotations/mpii_train.json',
            img_prefix=f'{mpii_data_root}/images/',
            data_cfg=mpii_data_cfg,
            pipeline=mpii_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='TopDownMpii2CocoDataset',
            ann_file=f'{mpii_data_root}/annotations/mpii_val.json',
            img_prefix=f'{mpii_data_root}/images/',
            data_cfg=mpii_data_cfg,
            pipeline=mpii_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='TopDownCrowdPose2CocoDataset',
            ann_file=f'{crowdpose_data_root}/annotations/mmpose_crowdpose_trainval.json',
            img_prefix=f'{crowdpose_data_root}/images/',
            data_cfg=crowdpose_data_cfg,
            pipeline=crowdpose_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='TopDownAic2CocoDataset',
            ann_file=f'{aic_data_root}/annotations/aic_train.json',
            img_prefix=f'{aic_data_root}/ai_challenger_keypoint_train_20170909/'
            'keypoint_train_images_20170902/',
            data_cfg=aic_data_cfg,
            pipeline=aic_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
        dict(
            type='TopDownAic2CocoDataset',
            ann_file=f'{aic_data_root}/annotations/aic_val.json',
            img_prefix=f'{aic_data_root}/ai_challenger_keypoint_validation_20170911/'
            'keypoint_validation_images_20170911/',
            data_cfg=aic_data_cfg,
            pipeline=aic_train_pipeline,
            dataset_info={{_base_.dataset_info}}),
    ],
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

## scale = downsample ratio from image resolution to heatmap resolution. 3.5 for EVA models, 4 for ViT models
custom_hooks = [
    dict(type='VisualizeHook', vis_every_iters=vis_every_iters, max_samples=16, scale=3.5),
]

workflow = [('train', evaluate_every_n_epochs), ('val', 1)] ## train 2 epochs, val 1 epoch