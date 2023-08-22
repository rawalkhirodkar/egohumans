import sys
import os
from mmpose.analysis.evaluation import sort_image_ap

# -----------------------------------
if len(sys.argv) < 2:
    print('Usage python quantitative_analysis <gt_file> <dt_file> <output_dir>')
    exit()

gt_file = sys.argv[1]
dt_file = sys.argv[2]
output_dir = sys.argv[3]

## make sure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sort_image_ap(gt_file=gt_file, dt_file=dt_file, output_dir=output_dir)