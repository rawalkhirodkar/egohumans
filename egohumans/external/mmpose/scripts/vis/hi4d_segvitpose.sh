cd ../..

###--------------------------------------------------------------
DEVICES=1,
RUN_FILE='./tools/seg_vitpose/demo.py'
RUN_MASK_FILE='./tools/seg_vitpose/draw_seg_bbox.py'

CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seg_ViTPose_large_coco_256x192.py'
CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/seg_checkpoints/large_best_AP_epoch_20.pth'

LINE_THICKNESS=3 ## 1 is default

DATA_ROOT=/media/rawalk/disk2/rawalk/datasets/hi4d/data
###--------------------------------------------------------------
PAIR=pair00
ACTION=dance00

# CAMERA=4
CAMERA=76


TIME_STAMP=60


##------------------------------------------------------------------------------
## make time stamp 5 digit
TIME_STAMP=$(printf "%06d" $TIME_STAMP)

###--------------------------------------------------------------
IMAGE=$DATA_ROOT/$PAIR/$ACTION/images/$CAMERA/$TIME_STAMP.jpg
BINARY_MASK='/home/rawalk/Desktop/ego/vitpose/tools/seg_vitpose/hi4d_images/'$PAIR'_'$ACTION'_'$CAMERA/$TIME_STAMP'_mask.jpg'
OVERLAY_IMAGE='/home/rawalk/Desktop/ego/vitpose/tools/seg_vitpose/hi4d_images/'$PAIR'_'$ACTION'_'$CAMERA/$TIME_STAMP'_overlay.jpg'

OUTPUT_DIR='/home/rawalk/Desktop/ego/vitpose/tools/seg_vitpose/hi4d_images/'$PAIR'_'$ACTION'_'$CAMERA

###--------------------------------------------------------------
python ${RUN_MASK_FILE} ${IMAGE} ${BINARY_MASK} ${OVERLAY_IMAGE}

# ###--------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_FILE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --img ${IMAGE} \
    --mask-path ${BINARY_MASK} \
    --thickness ${LINE_THICKNESS} \
    --out-img-root ${OUTPUT_DIR} \