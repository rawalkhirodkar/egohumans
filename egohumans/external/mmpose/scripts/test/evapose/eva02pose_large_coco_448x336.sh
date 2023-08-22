cd ../../..

## need to export NCCL_P2P_DISABLE=1 to make it work on A6000 gpus. export to bashrc

###--------------------------------------------------------------
# DEVICES=0,1,
# DEVICES=0,1,2,3,
DEVICES=0,1,2,3,4,5,6,7,


RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

###--------------------------------------------------------------
MODEL='eva02pose_large_coco_448x336'

## changin batch size is affecting the results! TODO: check why
# TEST_BATCH_SIZE_PER_GPU=16
TEST_BATCH_SIZE_PER_GPU=32 # default
# TEST_BATCH_SIZE_PER_GPU=64 
# TEST_BATCH_SIZE_PER_GPU=256

# CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/evapose_checkpoints/large_best_AP_epoch_10.pth'
# CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/evapose_checkpoints/large_all_datasets_best_AP_epoch_4.pth'
CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/evapose_checkpoints/evapose_448x336_all_datasets_new_pretrained_best_AP_epoch_13.pth'

# ## val set
# USE_GT_BBOX=True; EVAL_SET='val2017'; DETECTION_AP=0
# # USE_GT_BBOX=False; EVAL_SET='val2017'; DETECTION_AP=56 ## default for val
# # USE_GT_BBOX=False; EVAL_SET='val2017'; DETECTION_AP=67
USE_GT_BBOX=False; EVAL_SET='val2017'; DETECTION_AP=70 ## best for val

## test-dev set
# USE_GT_BBOX=False; EVAL_SET='test-dev2017'; DETECTION_AP=609 ## default for test set, 378148 bboxes
# USE_GT_BBOX=False; EVAL_SET='test-dev2017'; DETECTION_AP=0 ## best for test set by eva02

##--------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------
BBOX_DETECTION_FILE=data/coco/person_detection_results/COCO_${EVAL_SET}_detections_AP_H_${DETECTION_AP}_person.json
CONFIG_FILE=configs/body/2d_kpt_sview_rgb_img/custom_topdown_heatmap/coco/${MODEL}.py
OUTPUT_DIR=/home/rawalk/Desktop/ego/vitpose/Outputs/test/${MODEL}/${EVAL_SET}_bbox_${USE_GT_BBOX}_AP_${DETECTION_AP}

if [ "$EVAL_SET" = "val2017" ]; then
    ANNOTATION_FILE=data/coco/annotations/person_keypoints_${EVAL_SET}.json
    IMG_PREFIX=data/coco/${EVAL_SET}/

else
    ANNOTATION_FILE=data/coco/annotations/image_info_test-dev2017.json
    IMG_PREFIX=data/coco/test2017/
fi

###--------------------------------------------------------------
## set the options for the test
OPTIONS="$(echo "data.samples_per_gpu=$TEST_BATCH_SIZE_PER_GPU data.test.data_cfg.use_gt_bbox=${USE_GT_BBOX} 
    data.val_dataloader.samples_per_gpu=${TEST_BATCH_SIZE_PER_GPU} data.test_dataloader.samples_per_gpu=${TEST_BATCH_SIZE_PER_GPU}  
    data.test.ann_file=$ANNOTATION_FILE data.test.img_prefix=${IMG_PREFIX} 
    data.test.data_cfg.bbox_file=${BBOX_DETECTION_FILE}")"

# # ##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TEST_BATCH_SIZE_PER_GPU=8 ## works for single gpu

    OPTIONS="$(echo "data.samples_per_gpu=$TEST_BATCH_SIZE_PER_GPU data.workers_per_gpu=0 data.test.data_cfg.use_gt_bbox=${USE_GT_BBOX} \
    data.test.ann_file=$ANNOTATION_FILE data.test.img_prefix=${IMG_PREFIX} \
    data.test.data_cfg.bbox_file=${BBOX_DETECTION_FILE}")"

    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test.py ${CONFIG_FILE} $CHECKPOINT --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} $CHECKPOINT\
                ${NUM_GPUS} \
                --work-dir ${OUTPUT_DIR} \
                --cfg-options ${OPTIONS} \
                | tee ${LOG_FILE}
fi
