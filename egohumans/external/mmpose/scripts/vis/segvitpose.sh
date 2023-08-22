cd ../..

###--------------------------------------------------------------
DEVICES=1,
RUN_FILE='./tools/seg_vitpose/demo.py'
RUN_MASK_FILE='./tools/seg_vitpose/draw_seg_bbox.py'

# CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seg_ViTPose_base_coco_256x192.py'
# CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/seg_checkpoints/base_epoch_210.pth'

CONFIG_FILE='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seg_ViTPose_large_coco_256x192.py'
CHECKPOINT='/media/rawalk/disk1/rawalk/vitpose/seg_checkpoints/large_best_AP_epoch_20.pth'

LINE_THICKNESS=3 ## 1 is default

###--------------------------------------------------------------
# CAMERA_NAME='cam13'; TIME_STAMP='00050'
# CAMERA_NAME='cam04'; TIME_STAMP='00050'
# CAMERA_NAME='cam01'; TIME_STAMP='00050'
CAMERA_NAME='cam06'; TIME_STAMP='00050'

###--------------------------------------------------------------
IMAGE='/home/rawalk/Desktop/ego/vitpose/tools/seg_vitpose/images/'$CAMERA_NAME/$TIME_STAMP'.jpg'
BINARY_MASK='/home/rawalk/Desktop/ego/vitpose/tools/seg_vitpose/images/'$CAMERA_NAME/$TIME_STAMP'_mask.jpg'
OVERLAY_IMAGE='/home/rawalk/Desktop/ego/vitpose/tools/seg_vitpose/images/'$CAMERA_NAME/$TIME_STAMP'_overlay.jpg'

OUTPUT_DIR='/home/rawalk/Desktop/ego/vitpose/tools/seg_vitpose/images/'$CAMERA_NAME

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