cd ../../egohumans

###----------------------------------------------------------------
RUN_FILE='tools/vis/smpl.py'
SEQUENCE_ROOT_DIR='~/Desktop/egohumans/data'

###------------------------------------------------------------------
# BIG_SEQUENCE='01_tagging'; SEQUENCE='001_tagging'; DEVICES=0,
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='001_legoassemble'; DEVICES=0,
# BIG_SEQUENCE='03_fencing'; SEQUENCE='001_fencing'; DEVICES=0,
BIG_SEQUENCE='07_tennis'; SEQUENCE='001_tennis'; DEVICES=0,

###------------------------------------------------------------------
OUTPUT_DIR=$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/processed_data
mkdir -p ${OUTPUT_DIR};
SEQUENCE_PATH=$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE

###--------------------------visualize SMPL meshes-----------------------------
START_TIME=1
END_TIME=-1

CUDA_VISIBLE_DEVICES=${DEVICES} python $RUN_FILE \
                    --sequence_path ${SEQUENCE_PATH} \
                    --output_path $OUTPUT_DIR \
                    --start_time $START_TIME \
                    --end_time $END_TIME \


