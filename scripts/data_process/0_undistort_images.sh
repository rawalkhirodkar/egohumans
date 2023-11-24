cd ../../egohumans

###----------------------------------------------------------------
RUN_FILE='tools/data_process/undistort_images.py'
# SEQUENCE_ROOT_DIR='~/Desktop/egohumans/data'
SEQUENCE_ROOT_DIR='/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'

###------------------------------------------------------------------
## The variable $MODE can be either ['all', 'exo', 'ego']. To undistort all, exo or ego cameras.

# BIG_SEQUENCE='01_tagging'; SEQUENCE='001_tagging'; DEVICES=0; MODE='all'; 
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='001_legoassemble'; DEVICES=0; MODE='all';
# BIG_SEQUENCE='03_fencing'; SEQUENCE='001_fencing'; DEVICES=0; MODE='ego';
# BIG_SEQUENCE='07_tennis'; SEQUENCE='001_tennis'; DEVICES=0; MODE='exo';

###-----------------------------------------------------------------
# BIG_SEQUENCE='01_tagging'; SEQUENCE='001_tagging'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='002_tagging'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='003_tagging'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='004_tagging'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='005_tagging'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='006_tagging'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='007_tagging'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='008_tagging'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='009_tagging'; DEVICES=2; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='010_tagging'; DEVICES=2; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='011_tagging'; DEVICES=2; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='012_tagging'; DEVICES=2; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='013_tagging'; DEVICES=3; MODE='ego'; 
# BIG_SEQUENCE='01_tagging'; SEQUENCE='014_tagging'; DEVICES=3; MODE='ego'; 

##-----------------------------------------------------------------
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='001_legoassemble'; DEVICES=3; MODE='ego'; 
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='002_legoassemble'; DEVICES=3; MODE='ego'; 
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='003_legoassemble'; DEVICES=3; MODE='ego'; 
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='004_legoassemble'; DEVICES=3; MODE='ego'; 
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='005_legoassemble'; DEVICES=3; MODE='ego'; 
# BIG_SEQUENCE='02_legoassemble'; SEQUENCE='006_legoassemble'; DEVICES=3; MODE='ego'; 

# ##-----------------------------------------------------------------
# BIG_SEQUENCE='03_fencing'; SEQUENCE='001_fencing'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='002_fencing'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='003_fencing'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='004_fencing'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='005_fencing'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='006_fencing'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='007_fencing'; DEVICES=2; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='008_fencing'; DEVICES=2; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='009_fencing'; DEVICES=2; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='010_fencing'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='011_fencing'; DEVICES=0; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='012_fencing'; DEVICES=1; MODE='ego'; 
# BIG_SEQUENCE='03_fencing'; SEQUENCE='013_fencing'; DEVICES=1; MODE='ego'; 
BIG_SEQUENCE='03_fencing'; SEQUENCE='014_fencing'; DEVICES=2; MODE='ego'; 


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

