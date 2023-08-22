#!/bin/bash

FOLDER_NAME="hi4d_pair00_dance"

bbox_dir="$HOME/Desktop/ego/cliff/data/test_samples/$FOLDER_NAME/bbox/"
front_dir="$HOME/Desktop/ego/cliff/data/test_samples/$FOLDER_NAME/front_view_hr48/"
side_dir="$HOME/Desktop/ego/cliff/data/test_samples/$FOLDER_NAME/side_view_hr48/"
output_dir="$HOME/Desktop/ego/cliff/data/test_samples/$FOLDER_NAME/combine/"

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
