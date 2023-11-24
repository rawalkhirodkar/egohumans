cd ../../../egohumans/external/midas

source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 
conda activate midas

###----------------------------------------------------------------
RUN_FILE='tools/create_benchmark/coco_video_format.py'

# SEQUENCE_ROOT_DIR='~/Desktop/egohumans/data'
SEQUENCE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'

# ###------------------------------------------------------------------
# BIG_SEQUENCE='01_tagging'

# SEQUENCE='001_tagging'; DEVICES=0,
# SEQUENCE='002_tagging'; DEVICES=0,
# SEQUENCE='003_tagging'; DEVICES=0,
# SEQUENCE='004_tagging'; DEVICES=0,
# SEQUENCE='005_tagging'; DEVICES=1,
# SEQUENCE='006_tagging'; DEVICES=1,
# SEQUENCE='007_tagging'; DEVICES=1,
# SEQUENCE='008_tagging'; DEVICES=1,
# SEQUENCE='009_tagging'; DEVICES=2,
# SEQUENCE='010_tagging'; DEVICES=2,
# SEQUENCE='011_tagging'; DEVICES=2,
# SEQUENCE='012_tagging'; DEVICES=2,
# SEQUENCE='013_tagging'; DEVICES=3,
# SEQUENCE='014_tagging'; DEVICES=3,

# ###------------------------------------------------------------------
# BIG_SEQUENCE='02_legoassemble'
# 
# SEQUENCE='001_legoassemble'; DEVICES=3,
# SEQUENCE='002_legoassemble'; DEVICES=3,
# SEQUENCE='003_legoassemble'; DEVICES=0,
# SEQUENCE='004_legoassemble'; DEVICES=0, 
# SEQUENCE='005_legoassemble'; DEVICES=1, 
# SEQUENCE='006_legoassemble'; DEVICES=1,

# ###------------------------------------------------------------------
BIG_SEQUENCE='03_fencing'

# SEQUENCE='001_fencing'; DEVICES=2,
# SEQUENCE='002_fencing'; DEVICES=3, 
# SEQUENCE='003_fencing'; DEVICES=0, 
# SEQUENCE='004_fencing'; DEVICES=3,
# SEQUENCE='005_fencing'; DEVICES=0, 
# SEQUENCE='006_fencing'; DEVICES=2,
# SEQUENCE='007_fencing'; DEVICES=3,
# SEQUENCE='008_fencing'; DEVICES=1, 
SEQUENCE='009_fencing'; DEVICES=2,
# SEQUENCE='010_fencing'; DEVICES=1, 
# SEQUENCE='011_fencing'; DEVICES=3,
# SEQUENCE='012_fencing'; DEVICES=3,
# SEQUENCE='013_fencing'; DEVICES=1, 
# SEQUENCE='014_fencing'; DEVICES=1,

###-----------------------------------------------------------------
INPUT_DIR=$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE
OUTPUT_DIR=$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE

CUDA_VISIBLE_DEVICES=${DEVICES} python run_ego_exo.py \
                    --model_type dpt_large \
                    --input_path $INPUT_DIR \
                    --output_path $OUTPUT_DIR