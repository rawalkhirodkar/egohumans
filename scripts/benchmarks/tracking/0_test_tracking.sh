cd ../../../egohumans/external/mmtracking

# ###----------------------------------------------------------------
RUN_FILE='./tools/dist_test.sh'
RUN_SINGLE_GPU_FILE='./tools/test.py'
RUN_VIS_FILE='./demo/demo_mot_vis.py'

SEQUENCE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/benchmark'
SEQUENCE_IMAGE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'

##------------------------------------------------------------------
SAVE_BIG_SEQUENCE_NAME='01_tagging:02_legoassemble:03_fencing' ## save dir name for the annotations
SEQUENCES='all'; DEVICES=0,1,

# ##------------------------------------------------------------------
# SAVE_BIG_SEQUENCE_NAME='01_tagging' ## save dir name for the annotations
# SEQUENCES='001_tagging'; DEVICES=0,1,

####--------------------------------------------------------------------
USE_GT_BBOX=false;
# METHOD='sort'
# METHOD='deepsort'
# METHOD='qdtrack'
# METHOD='tracktor'
# METHOD='ocsort' 
# METHOD='bytetrack'
# METHOD='simplebaseline'
METHOD='egoformer'

# ####--------------------------------------------------------------------
# USE_GT_BBOX=true;
# METHOD='sort'
# METHOD='deepsort'
# METHOD='qdtrack'
# METHOD='tracktor'
# METHOD='ocsort'
# METHOD='bytetrack'
# METHOD='simplebaseline'
# METHOD='egoformer'

####--------------------------------------------------------------------
# MODE='exo'
MODE='ego_rgb'
# MODE='ego_slam'

EVAL_TYPE='eval' ## pure evaluation
# EVAL_TYPE='debug' ## debug
# EVAL_TYPE='vis' ## for visualization

###------------------------------------------------------------------
## public: using detections in the det file, private: using offshelf model 

## bytetrack
BYTETRACK_PUBLIC_CONFIG_FILE='configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-public-half.py'
BYTETRACK_PRIVATE_CONFIG_FILE='configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
BYTETRACK_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.pth'

## ocsort
OCSORT_PUBLIC_CONFIG_FILE=''
OCSORT_PRIVATE_CONFIG_FILE='configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py'
OCSORT_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.pth'

# ## tracktor
TRACKTOR_PUBLIC_CONFIG_FILE='configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8e_mot20-public-half.py'
TRACKTOR_PRIVATE_CONFIG_FILE='configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8e_mot20-private-half.py'
TRACKTOR_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/tracktor/faster-rcnn_r50_fpn_8e_mot20-half.pth' ## this doesnt matter, it will download from the internet

# ## deepsort
DEEPSORT_PUBLIC_CONFIG_FILE='configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-public-half.py'
DEEPSORT_PRIVATE_CONFIG_FILE='configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
DEEPSORT_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/deepsort/faster-rcnn_r50_fpn_4e_mot17-half.pth' 

# ## sort
SORT_PUBLIC_CONFIG_FILE='configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-public-half.py'
SORT_PRIVATE_CONFIG_FILE='configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private-half.py'
SORT_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/sort/faster-rcnn_r50_fpn_4e_mot17-half.pth' 

## qdtrack
QDTRACK_PUBLIC_CONFIG_FILE=''
QDTRACK_PRIVATE_CONFIG_FILE='configs/mot/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17-private-half.py'
QDTRACK_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/qdtrack/qdtrack_faster-rcnn_r50_fpn_4e_mot17.pth' 

# ###------------------------------------------------------------------
# # # # ### custom, our method with 3d kalman filter
SIMPLEBASELINE_PUBLIC_CONFIG_FILE='configs/mot/custom_bytetrack/bytetrack_yolox_x_crowdhuman_mot17-public-half.py'
SIMPLEBASELINE_PRIVATE_CONFIG_FILE='configs/mot/custom_bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
SIMPLEBASELINE_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.pth'

EGOFORMER_PUBLIC_CONFIG_FILE='configs/mot/custom_bytetrack/bytetrack_yolox_x_crowdhuman_mot17-public-half.py'
EGOFORMER_PRIVATE_CONFIG_FILE='configs/mot/custom_egoformer/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
EGOFORMER_CHECKPOINT='/home/rawalk/Desktop/ego/mmtracking/checkpoints/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.pth'

##-------------------------------------------------------------------
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
BATCH_SIZE=8 ## batch size has not effect for tracking, default is 1

ANNOTATION_FILE=$SEQUENCE_ROOT_DIR/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/coco_track/tracking_$MODE.json
GT_DETECTION_FILE=$SEQUENCE_ROOT_DIR/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/coco_track/detections_$MODE.pkl ## ground truth bbox and 3D using midas

#----------------------------------------------------------------------
if [[ $METHOD == "egoformer" ]]
then
    PUBLIC_CONFIG_FILE=$EGOFORMER_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$EGOFORMER_PRIVATE_CONFIG_FILE
    CHECKPOINT=$EGOFORMER_CHECKPOINT

    GT_DETECTION_FILE=$SEQUENCE_ROOT_DIR/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/coco_track/egoformer_detections_$MODE.pkl
    EGOFORMER_YOLOX_DETECTION_FILE=$SEQUENCE_ROOT_DIR/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/coco_track/egoformer_yolox_detections_$MODE.pkl ## use yolox detections

elif [[ $METHOD == "simplebaseline" ]]
then
    PUBLIC_CONFIG_FILE=$SIMPLEBASELINE_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$SIMPLEBASELINE_PRIVATE_CONFIG_FILE
    CHECKPOINT=$SIMPLEBASELINE_CHECKPOINT

elif [[ $METHOD == "sort" ]]
then
    PUBLIC_CONFIG_FILE=$SORT_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$SORT_PRIVATE_CONFIG_FILE
    CHECKPOINT=$SORT_CHECKPOINT

elif [[ $METHOD == "deepsort" ]]
then
    PUBLIC_CONFIG_FILE=$DEEPSORT_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$DEEPSORT_PRIVATE_CONFIG_FILE
    CHECKPOINT=$DEEPSORT_CHECKPOINT

elif [[ $METHOD == "qdtrack" ]]
then
    PUBLIC_CONFIG_FILE=$QDTRACK_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$QDTRACK_PRIVATE_CONFIG_FILE
    CHECKPOINT=$QDTRACK_CHECKPOINT

elif [[ $METHOD == "tracktor" ]]
then
    PUBLIC_CONFIG_FILE=$TRACKTOR_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$TRACKTOR_PRIVATE_CONFIG_FILE
    CHECKPOINT=$TRACKTOR_CHECKPOINT

elif [[ $METHOD == "ocsort" ]]
then
    PUBLIC_CONFIG_FILE=$OCSORT_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$OCSORT_PRIVATE_CONFIG_FILE
    CHECKPOINT=$OCSORT_CHECKPOINT

elif [[ $METHOD == "bytetrack" ]]
then
    PUBLIC_CONFIG_FILE=$BYTETRACK_PUBLIC_CONFIG_FILE
    PRIVATE_CONFIG_FILE=$BYTETRACK_PRIVATE_CONFIG_FILE
    CHECKPOINT=$BYTETRACK_CHECKPOINT

fi


#----------------------------------------------------------------------
if $USE_GT_BBOX
then
    OPTIONS="$(echo "data.test.img_prefix=$SEQUENCE_IMAGE_ROOT_DIR/ data.test.ann_file=${ANNOTATION_FILE} data.test.detection_file=$GT_DETECTION_FILE")"
    OUTPUT_DIR=$SEQUENCE_ROOT_DIR/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/"output"/"tracking"/$METHOD/$MODE"_gt_bbox"
    CONFIG_FILE=$PUBLIC_CONFIG_FILE

else
    OPTIONS="$(echo "data.test.img_prefix=$SEQUENCE_IMAGE_ROOT_DIR/ data.test.ann_file=${ANNOTATION_FILE} data.test.detection_file=None")"
    OUTPUT_DIR=$SEQUENCE_ROOT_DIR/$SAVE_BIG_SEQUENCE_NAME/$SEQUENCES/"output"/"tracking"/$METHOD/$MODE"_det_bbox"
    CONFIG_FILE=$PRIVATE_CONFIG_FILE


    if [[ $METHOD == "egoformer" ]]
    then
        OPTIONS="$(echo "${OPTIONS} data.test.egoformer_yolox_file=${EGOFORMER_YOLOX_DETECTION_FILE}")"
    fi

fi

OUTPUT_FILE=${OUTPUT_DIR}/'coco_track.pkl'

##----------------------------------------------------------------
mkdir -p ${OUTPUT_DIR}; 

##-----------------------------------------------
if [[ $EVAL_TYPE == "eval" ]]
then

    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))
    OPTIONS="$(echo "${OPTIONS} data.samples_per_gpu=${BATCH_SIZE}")"

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} $NUM_GPUS \
        --work-dir $OUTPUT_DIR \
        --cfg-options $OPTIONS \
        --out $OUTPUT_FILE \
        --eval track \
        --checkpoint $CHECKPOINT 

elif [[ $EVAL_TYPE == "debug" ]]
then
    ##----------------------------single gpu eval and debug-----------------------
    OPTIONS="$(echo "${OPTIONS} data.samples_per_gpu=2 data.workers_per_gpu=0")"

    CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_SINGLE_GPU_FILE} $CONFIG_FILE \
             --work-dir $OUTPUT_DIR \
            --cfg-options $OPTIONS \
            --out $OUTPUT_FILE \
            --eval track \
            --checkpoint $CHECKPOINT 

elif [[ $EVAL_TYPE == "vis" ]]
then    
    ##----for visualization, single gpu------------------
    OPTIONS="$(echo "${OPTIONS} data.samples_per_gpu=${BATCH_SIZE}")"
    CUDA_VISIBLE_DEVICES=${DEVICES} python ${RUN_SINGLE_GPU_FILE} $CONFIG_FILE \
             --work-dir $OUTPUT_DIR \
            --cfg-options $OPTIONS \
            --out $OUTPUT_FILE \
            --eval track \
            --checkpoint $CHECKPOINT \
            --show-dir ${OUTPUT_DIR}

fi

echo 'method:' $METHOD
echo 'config: ' $CONFIG_FILE
echo 'checkpoint:' $CHECKPOINT
