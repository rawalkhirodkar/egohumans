cd ../../../..

# Model Details
# Model Type: Image classification / feature backbone
# Model Stats:
# Params (M): 326.4
# GMACs: 362.4
# Activations (M): 690.0
# Image size: 448 x 448

###--------------------------------------------------------------
# DEVICES=2,1,
DEVICES=0,1,2,3,4,5,
# DEVICES=0,1,2,
# DEVICES=4,5,6,7,
# DEVICES=2,3,4,5,6,7,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# ###----------------------------------------------------
MODEL='judge_eva02pose_large_coco_448x336'
# TRAIN_BATCH_SIZE_PER_GPU=10 ### default is 64
TRAIN_BATCH_SIZE_PER_GPU=20 ### default is 64

PRETRAINED='/media/rawalk/disk1/rawalk/vitpose/pretrained/eva02_large_patch14_448.mim_in22k_ft_in22k.pth'
RESUME_FROM=''

POSE_MODEL_CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/evapose_checkpoints/evapose_448x336_all_datasets_new_pretrained_best_AP_epoch_13.pth'

###--------------------------------------------------------------
JUDGE_KEYPOINT_INDEX=0; JUDGE_KEYPOINT_NAME='nose'; DEVICES=0,1,2,5,
# JUDGE_KEYPOINT_INDEX=1; JUDGE_KEYPOINT_NAME='left_eye'; DEVICES=1,
# JUDGE_KEYPOINT_INDEX=2; JUDGE_KEYPOINT_NAME='right_eye'; DEVICES=2,
# JUDGE_KEYPOINT_INDEX=3; JUDGE_KEYPOINT_NAME='left_ear'; DEVICES=3,
# JUDGE_KEYPOINT_INDEX=4; JUDGE_KEYPOINT_NAME='right_ear'; DEVICES=0,
# JUDGE_KEYPOINT_INDEX=5; JUDGE_KEYPOINT_NAME='left_shoulder'; DEVICES=5,
# JUDGE_KEYPOINT_INDEX=6; JUDGE_KEYPOINT_NAME='right_shoulder'; DEVICES=1,
# JUDGE_KEYPOINT_INDEX=7; JUDGE_KEYPOINT_NAME='left_elbow'; DEVICES=2,
# JUDGE_KEYPOINT_INDEX=8; JUDGE_KEYPOINT_NAME='right_elbow'; DEVICES=3,
# JUDGE_KEYPOINT_INDEX=9; JUDGE_KEYPOINT_NAME='left_wrist'
# JUDGE_KEYPOINT_INDEX=10; JUDGE_KEYPOINT_NAME='right_wrist'
# JUDGE_KEYPOINT_INDEX=11; JUDGE_KEYPOINT_NAME='left_hip'
# JUDGE_KEYPOINT_INDEX=12; JUDGE_KEYPOINT_NAME='right_hip'
# JUDGE_KEYPOINT_INDEX=13; JUDGE_KEYPOINT_NAME='left_knee'
# JUDGE_KEYPOINT_INDEX=14; JUDGE_KEYPOINT_NAME='right_knee'
# JUDGE_KEYPOINT_INDEX=15; JUDGE_KEYPOINT_NAME='left_ankle'
# JUDGE_KEYPOINT_INDEX=16; JUDGE_KEYPOINT_NAME='right_ankle'


##--------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------
CONFIG_FILE=configs/body/2d_kpt_sview_rgb_img/judge_topdown/coco/${MODEL}.py

## the judge keypoint index use in name, zero fill to 2 digits
OUTPUT_DIR='Outputs/train/'${MODEL}/$(printf "%02d" ${JUDGE_KEYPOINT_INDEX})_${JUDGE_KEYPOINT_NAME}
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

###--------------------------------------------------------------
if [ "$PRETRAINED" != "" ]; then
    OPTIONS="$(echo "model.pretrained=$PRETRAINED data.samples_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU \
                    model.judge_keypoint_idx=${JUDGE_KEYPOINT_INDEX} \
                    model.pose_model_checkpoint=${POSE_MODEL_CHECKPOINT}")"
else
    OPTIONS="$(echo "data.samples_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU \
                    model.judge_keypoint_idx=${JUDGE_KEYPOINT_INDEX} \
                    model.pose_model_checkpoint=${POSE_MODEL_CHECKPOINT}")"
fi

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TRAIN_BATCH_SIZE_PER_GPU=8 ## works for single gpu
    OPTIONS="$(echo "model.pretrained=$PRETRAINED data.samples_per_gpu=${TRAIN_BATCH_SIZE_PER_GPU} data.workers_per_gpu=0 \
                    model.judge_keypoint_idx=${JUDGE_KEYPOINT_INDEX} \
                     model.pose_model_checkpoint=${POSE_MODEL_CHECKPOINT}")"

    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/train.py ${CONFIG_FILE} --work-dir ${OUTPUT_DIR} --no-validate --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))
    SEED='0'

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}


    ## if RESUME_FROM is not '', then resume training from the given checkpoint. Else run the command without resume-from
    if [ "$RESUME_FROM" != "" ]; then
        CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} \
                ${NUM_GPUS} \
                --work-dir ${OUTPUT_DIR} \
                --seed ${SEED} \
                --cfg-options ${OPTIONS} \
                --resume-from ${RESUME_FROM} \
                | tee ${LOG_FILE}
    else
        CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} \
                ${NUM_GPUS} \
                --work-dir ${OUTPUT_DIR} \
                --seed ${SEED} \
                --cfg-options ${OPTIONS} \
                | tee ${LOG_FILE}
    fi
    
fi

