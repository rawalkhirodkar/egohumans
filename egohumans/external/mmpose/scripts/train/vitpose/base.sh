cd ../../..

## need to export NCCL_P2P_DISABLE=1 to make it work on A6000 gpus. export to bashrc

###--------------------------------------------------------------
# DEVICES=0,1,
# DEVICES=0,1,2,3,4,5,6,7,
DEVICES=0,1,2,3,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

# # ###-----------------------------w32 256 x 192---------------------------
# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py'
# PRETRAINED='/media/rawalk/disk1/rawalk/vitpose/pretrained/mae_pretrain_vit_base.pth'
# # OUTPUT_DIR='Outputs/train/ViTPose_base_coco_256x192'

# ###-----------------------------w32 256 x 192---------------------------
CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/gtbbox_ViTPose_base_coco_256x192.py'
PRETRAINED='/media/rawalk/disk1/rawalk/vitpose/pretrained/mae_pretrain_vit_base.pth'
OUTPUT_DIR='Outputs/train/gtbbox_ViTPose_base_coco_256x192'

TRAIN_BATCH_SIZE_PER_GPU=64 ## default

###--------------------------------------------------------------
OPTIONS="$(echo " model.pretrained=$PRETRAINED data.samples_per_gpu=$TRAIN_BATCH_SIZE_PER_GPU")"

# #####---------------------multi-gpu training---------------------------------
NUM_GPUS_STRING_LEN=${#DEVICES}
NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))
SEED='0'

LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} \
			${NUM_GPUS} \
			--work-dir ${OUTPUT_DIR} \
			--seed ${SEED} \
			--cfg-options ${OPTIONS} \
			| tee ${LOG_FILE}


# #####---------------------debugging on a single gpu---------------------------------
# TRAIN_BATCH_SIZE_PER_GPU=8 ## works for single gpu
# OPTIONS="$(echo "data.samples_per_gpu=${TRAIN_BATCH_SIZE_PER_GPU} data.workers_per_gpu=0")"

# CUDA_VISIBLE_DEVICES=${DEVICES} python tools/train.py ${CONFIG_FILE} --work-dir ${OUTPUT_DIR} --no-validate --cfg-options ${OPTIONS} \
