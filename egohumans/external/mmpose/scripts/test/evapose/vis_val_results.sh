cd ../../..

# # # # -----------------------------------------------------------
gt_file='data/coco/annotations/person_keypoints_val2017.json'


##--------------------------------------------------------------
MODEL='eva02pose_large_coco_448x336'

# ## val set
# USE_GT_BBOX=True; EVAL_SET='val2017'; DETECTION_AP=0
# # USE_GT_BBOX=False; EVAL_SET='val2017'; DETECTION_AP=56 ## default for val
# # USE_GT_BBOX=False; EVAL_SET='val2017'; DETECTION_AP=67
USE_GT_BBOX=False; EVAL_SET='val2017'; DETECTION_AP=70 ## best for val

## test-dev set
# USE_GT_BBOX=False; EVAL_SET='test-dev2017'; DETECTION_AP=609 ## default for test set, 378148 bboxes
# USE_GT_BBOX=False; EVAL_SET='test-dev2017'; DETECTION_AP=0 ## best for test set by eva02

OUTPUT_DIR=/home/rawalk/Desktop/ego/vitpose/Outputs/test/${MODEL}/${EVAL_SET}_bbox_${USE_GT_BBOX}_AP_${DETECTION_AP}
##--------------------------------------------------------------

dt_file=${OUTPUT_DIR}/result_keypoints.json

output_dir=Outputs/test/${MODEL}/${EVAL_SET}_bbox_${USE_GT_BBOX}_AP_${DETECTION_AP}

python tools/analysis/sort_image_ap.py $gt_file $dt_file $output_dir
