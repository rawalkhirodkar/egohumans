cd ../../egohumans

###----------------------------------------------------------------
RUN_FILE='tools/data_process/undistort_images.py'
# SEQUENCE_ROOT_DIR='~/Desktop/egohumans/data'
SEQUENCE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'


###------------------------------------------------------------------
## The variable $MODE can be either ['all', 'exo', 'ego']. To undistort all, exo or ego cameras.

BIG_SEQUENCE='01_tagging'; SEQUENCE='001_tagging'; DEVICES=0; MODE='all'; 
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='001_legoassemble'; DEVICES=0; MODE='all';
# BIG_SEQUENCE='03_fencing'; SEQUENCE='001_fencing'; DEVICES=0; MODE='ego';
# BIG_SEQUENCE='07_tennis'; SEQUENCE='001_tennis'; DEVICES=0; MODE='exo';

###------------------------------------------------------------------
OUTPUT_DIR=$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE
mkdir -p ${OUTPUT_DIR};
SEQUENCE_PATH=$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE

## if MODE variable is not set, then set it to 'all'
if [ -z ${MODE+x} ]; then MODE='all'; fi

# # # # # # # # ##-----------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICES} python $RUN_FILE \
                    --sequence_path ${SEQUENCE_PATH} \
                    --output_path $OUTPUT_DIR \
                    --mode $MODE \

