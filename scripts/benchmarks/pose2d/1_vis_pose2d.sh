cd ../../../egohumans/external/mmpose

# ###----------------------------------------------------------------
RUN_VIS_FILE='demo/custom_vis.py'

# ###--------------------------------------resnet50--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth'

# ###--------------------------------------resnet101--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth'

# # ###--------------------------------------resnet152--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth'

# # ###--------------------------------------hrnet32 udp--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp-aba0be42_20210220.pth'

# # ###--------------------------------------hrnet48 udp--------------------------------
CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_udp.py'
CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_256x192_udp-2554c524_20210223.pth'

# ###--------------------------------------swin-t--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_t_p4_w7_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/swin/swin_t_p4_w7_coco_256x192-eaefe010_20220503.pth'

# # ###--------------------------------------swin-b--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_b_p4_w7_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/swin/swin_b_p4_w7_coco_256x192-7432be9e_20220705.pth'

# # ###--------------------------------------swin-l--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_256x192-642a89db_20220705.pth'

# # ###--------------------------------------hrformer small--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_small_coco_256x192-5310d898_20220316.pth'


# ###--------------------------------------hrformer base--------------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_base_coco_256x192.py'
# CHECKPOINT='https://download.openmmlab.com/mmpose/top_down/hrformer/hrformer_base_coco_256x192-6f5f1169_20220316.pth'

# ###------------------------------------------------------------------
SEQUENCE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'
SAVE_BIG_SEQUENCE_NAME='01_tagging:02_legoassemble:03_fencing' ## save dir name for the annotations

DEVICES=0,1,2,3,
SEQUENCES='all'

###---------------------------------------------------------------------
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
BATCH_SIZE=16

## pick a eval mode
MODE='ego_rgb'
# MODE='ego_slam'
# MODE='exo'

ANNOTATION_FILE=$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/coco/person_keypoints_$MODE.json

#----------------------------------------------------------------------
OUTPUT_DIR=$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/"output"/"pose2d"/$MODE
OUTPUT_FILE=${OUTPUT_DIR}/'coco.yml'

OPTIONS="data.test.data_cfg.use_gt_bbox=True data.test.img_prefix=$SEQUENCE_ROOT_DIR/" ## if using gt bbox
# OPTIONS="data.test.data_cfg.use_gt_bbox=False data.test.img_prefix=''" ## if not using gt bbox

OPTIONS="$(echo "$OPTIONS data.test.ann_file=${ANNOTATION_FILE} data.samples_per_gpu=${BATCH_SIZE}")"

 ##----------------------------------------------------------------
mkdir -p ${OUTPUT_DIR}; 

##-----------------------------------------------
THICKNESS=3
RADIUS=4

CUDA_VISIBLE_DEVICES=${DEVICES} python $RUN_VIS_FILE ${CONFIG_FILE} ${CHECKPOINT} \
        --json-file $ANNOTATION_FILE \
        --img-root $SEQUENCE_ROOT_DIR \
        --out-img-root $OUTPUT_DIR \
        --radius $RADIUS \
        --thickness $THICKNESS \

echo $CONFIG_FILE
echo $CHECKPOINT
echo $MODE
