cd ../..

##------------------------------------------------------------------------------
PAIR=pair00
ACTION=dance00
# CAMERA=4
CAMERA=76

##------------------------------------------------------------------------------
DATA_ROOT=/media/rawalk/disk2/rawalk/datasets/hi4d

SOURCE_FOLDER=$DATA_ROOT/$PAIR/$ACTION/images/$CAMERA
TARGET_DIR=/home/rawalk/Desktop/ego/cliff/data/test_samples

TARGET_FOLDER=$TARGET_DIR/hi4d/$PAIR'_'$ACTION'_'$CAMERA/imgs

echo $SOURCE_FOLDER $TARGET_FOLDER

## create target folder
mkdir -p $TARGET_FOLDER

## copy images
cp $SOURCE_FOLDER/*.jpg $TARGET_FOLDER

echo "done copying images"

###----------------------------------------------------------------------------
## input folder is target folder but remove the last folder
INPUT_FOLDER=${TARGET_FOLDER%/*}

CKPT_PATH=data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
BACKBONE=hr48

export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0
python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
                --input_path ${INPUT_FOLDER} --input_type folder \
                --show_bbox --show_sideView --save_results

###----------------------------------------------------------------------------
## concatenate front and side view images
#!/bin/bash

bbox_dir="$INPUT_FOLDER/bbox/"
front_dir="$INPUT_FOLDER/front_view_hr48/"
side_dir="$INPUT_FOLDER/side_view_hr48/"
output_dir="$INPUT_FOLDER/combine/"

mkdir -p $output_dir  # create output directory if it doesn't exist

for bbox_file in $bbox_dir*; do
    base_name=$(basename $bbox_file)  # get base filename
    seq_number=${base_name:0:6}  # extract sequence number
    front_file=$front_dir$seq_number"_front_view_cliff_hr48.jpg"  # construct front file path
    side_file=$side_dir$seq_number"_side_view_cliff_hr48.jpg"  # construct side file path
    output_file=$output_dir$seq_number"_combined.jpg"  # construct output file path

    if [[ -f $front_file ]] && [[ -f $side_file ]]; then  # if corresponding front and side files exist
        convert $bbox_file $front_file $side_file +append $output_file  # concatenate images
    fi
done

## create video
frame_rate=20
ffmpeg -framerate $frame_rate -pattern_type glob -i $output_dir'*.jpg' -pix_fmt yuv420p $INPUT_FOLDER/$PAIR'_'$ACTION'_'$CAMERA.mp4
