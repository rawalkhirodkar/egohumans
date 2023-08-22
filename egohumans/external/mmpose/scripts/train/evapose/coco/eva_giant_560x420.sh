cd ../../../..


## Download AIC data from https://tianchi.aliyun.com/dataset/147359?t=1685141321976
## god bless this guy! Alibaba cloud: Kiran123


###--------------------------------------------------------------
# DEVICES=2,1,
# DEVICES=0,1,2,3,4,5,
# DEVICES=4,5,6,7,
DEVICES=0,1,2,3,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# ###----------------------------------------------------
MODEL='evapose_giant_coco_560x420'
TRAIN_BATCH_SIZE_PER_GPU=6 ### default is 64

PRETRAINED='/media/rawalk/disk1/rawalk/vitpose/pretrained/eva_21k_1k_560px_psz14_ema_89p7.pt'
RESUME_FROM=''

##--------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------
CONFIG_FILE=configs/body/2d_kpt_sview_rgb_img/custom_topdown_heatmap/coco/${MODEL}.py
OUTPUT_DIR='Outputs/train/'${MODEL}
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

###--------------------------------------------------------------
if [ "$PRETRAINED" != "" ]; then
    OPTIONS="$(echo "model.pretrained=$PRETRAINED data.samples_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU")"
else
    OPTIONS="$(echo "data.samples_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU")"
fi

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TRAIN_BATCH_SIZE_PER_GPU=8 ## works for single gpu
    OPTIONS="$(echo "model.pretrained=$PRETRAINED data.samples_per_gpu=${TRAIN_BATCH_SIZE_PER_GPU} data.workers_per_gpu=0")"

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

