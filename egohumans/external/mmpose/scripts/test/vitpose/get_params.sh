cd ../../..

###--------------------------------------------------------------
RUN_FILE='./tools/analysis/get_flops.py'

# MODEL='small' ## 24.29 M, 5.26 GFLOPs
# MODEL='base' ## 89.99 M, 17.85 GFLOPs
# MODEL='large' ## 308.54 M, 59.78 GFLOPs
MODEL='huge' ## 637.21 M, 122.85 GFLOPs

CONFIG_FILE=configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_${MODEL}_coco_256x192.py

###-------------------------------------------------------------- 
python $RUN_FILE $CONFIG_FILE --shape 256 192