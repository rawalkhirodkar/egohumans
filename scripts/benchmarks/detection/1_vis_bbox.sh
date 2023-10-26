cd ../../../egohumans/external/mmdetection


# ###----------------------------------------------------------------
RUN_VIS_FILE='./tools/test.py'

CONFIG_FILE='configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
CHECKPOINT='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

# ###------------------------------------------------------------------
SAVE_BIG_SEQUENCE_NAME='01_tagging:02_legoassemble:03_fencing' ## save dir name for the annotations
SEQUENCE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'

## pick a eval mode
# MODE='exo'
MODE='ego_rgb'
# MODE='ego_slam'

SEQUENCES='all'; DEVICES=0,1,

##-------------------------------------------------------------------
SEQUENCE_PATH=$SEQUENCE_ROOT_DIR/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES

PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
BATCH_SIZE=64

ANNOTATION_FILE=$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/coco/person_keypoints_$MODE.json

#----------------------------------------------------------------------
OUTPUT_DIR=$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/"output"/"bbox"/$MODE
OUTPUT_FILE=${OUTPUT_DIR}/'coco_bbox.pkl'
OPTIONS="data.test.img_prefix=$SEQUENCE_ROOT_DIR"
OPTIONS="$(echo "$OPTIONS data.test.ann_file=${ANNOTATION_FILE} data.samples_per_gpu=${BATCH_SIZE}")"

# # # # # # # # # ##-----------------------------------------------
NUM_GPUS_STRING_LEN=${#DEVICES}
NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

# # # ##----for visualization, single gpu------------------
CUDA_VISIBLE_DEVICES=${DEVICES} python $RUN_VIS_FILE ${CONFIG_FILE} ${CHECKPOINT} \
        --cfg-options $OPTIONS \
        --show-dir $OUTPUT_DIR \
        --eval 'bbox' \
