cd ../../../..

###--------------------------------------------------------------
# DEVICES=2,1,
DEVICES=0,1,2,3,4,5,
# DEVICES=2,3,4,5
# DEVICES=2,3,4,5,6,7,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# ###----------------------------------------------------
MODEL='decoder_eva02pose_large_coco+mpii+crowdpose+aic_448x336'
TRAIN_BATCH_SIZE_PER_GPU=26 ### default is 64

PRETRAINED='/media/rawalk/disk1/rawalk/vitpose/pretrained/eva02_large_patch14_448.mim_in22k_ft_in22k.pth'

RESUME_FROM=''
# RESUME_FROM='/home/rawalk/Desktop/ego/vitpose/Outputs/train/eva02pose_large_coco_448x336/05-19-2023_20:02:11/latest.pth'

##--------------------------------------------------------------
mode='debug'
# mode='multi-gpu'

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

if [ "$LOAD_FROM" != "" ]; then
    OPTIONS="$(echo "${OPTIONS} load_from=$LOAD_FROM")"
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

