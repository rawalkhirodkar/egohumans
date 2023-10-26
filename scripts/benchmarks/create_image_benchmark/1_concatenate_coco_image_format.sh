cd ../../../egohumans

###----------------------------------------------------------------
RUN_FILE='tools/create_benchmark/concatenate_coco_image_format.py'

# SEQUENCE_ROOT_DIR='~/Desktop/egohumans/data'
SEQUENCE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'

# # ##-----------------------first two subsequences of tagging----------------------------
# SAVE_BIG_SEQUENCE_NAME='01_tagging'
# BIG_SEQUENCES='01_tagging:01_tagging'
# SEQUENCES='001_tagging:002_tagging'; 
# DEVICES=0,

# ###-----------------test set for egohumans----------------------------
SAVE_BIG_SEQUENCE_NAME='01_tagging:02_legoassemble:03_fencing' ## save dir name for the annotations
DEVICES=0,

BIG_SEQUENCES='01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging:01_tagging'
SEQUENCES='001_tagging:002_tagging:003_tagging:004_tagging:005_tagging:006_tagging:007_tagging:008_tagging:009_tagging:010_tagging:011_tagging:012_tagging:013_tagging:014_tagging'

BIG_SEQUENCES+=':'
SEQUENCES+=':'
BIG_SEQUENCES+='02_legoassemble:02_legoassemble:02_legoassemble:02_legoassemble:02_legoassemble:02_legoassemble'
SEQUENCES+='001_legoassemble:002_legoassemble:003_legoassemble:004_legoassemble:005_legoassemble:006_legoassemble'

BIG_SEQUENCES+=':'
SEQUENCES+=':'
BIG_SEQUENCES+='03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing:03_fencing'
SEQUENCES+='001_fencing:002_fencing:003_fencing:004_fencing:005_fencing:006_fencing:003_fencing:008_fencing:009_fencing:010_fencing:011_fencing:012_fencing:013_fencing:014_fencing'

# # ###----------------------------------------------------------------
OUTPUT_DIR="$SEQUENCE_ROOT_DIR/benchmark/"$SAVE_BIG_SEQUENCE_NAME/"all"

mkdir -p ${OUTPUT_DIR}; 

# # # # # # # # ##-----------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICES} python $RUN_FILE \
                    --big_sequences ${BIG_SEQUENCES} \
                    --sequences ${SEQUENCES} \
                    --root_dir $SEQUENCE_ROOT_DIR \
                    --output_path $OUTPUT_DIR \
