# 2D Pose Estimation

We provide evaluation scripts to test 2D human pose estimation methods on the EgoHumans test set.
You can pick your favorite method from the mmpose model zoo and evaluate either on ```rgb-ego```, ```slam-ego``` (grayscale stereo) or ```rgb-exo``` images.


## Testing Pose Estimators

- Please refer to the test script under ```./scripts/benchmarks/pose2d/0_test_pose2d.sh```
- Modify the variables ```$CONFIG_FILE``` and ```$CHECKPOINT``` to select a detector. Refer to the [model zoo](https://github.com/rawalkhirodkar/egohumans/tree/main/egohumans/external/mmpose/configs) for supported methods.
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variable ```$SAVE_BIG_SEQUENCE_NAME``` to select the combination of big sequences used for evaluations.
- Modify the variable ```$MODE``` to be either ```ego_rgb```, ```ego_slam``` or ```exo```.
- Additionally you can set the number of gpus and batch size to be used during testing.
- The script assumes ground-truth bounding box for evaluation.
- Run the script
```shell
cd scripts/benchmarks/pose2d
chmod +x 0_test_pose2d.sh
./0_test_pose2d.sh
```
- The code will create a coco_bbox.pkl file under the folder ```$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/all/output/bbox/$MODE```.


## Results
We use ground-truth bounding boxes for evaluating the following methods. Click on the method name for the config file.

### Ego-RGB 
Total test images: 47740  
Total test instances: 78880


|Method                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                                            |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: |
| [SimpleBaseline-ResNet50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py) |  256x192   | 0.716 |      0.909      |      0.833      | 0.748 |      0.912      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth) |
| [SimpleBaseline-ResNet101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192.py) |  256x192   | 0.719 |      0.909      |      0.833      | 0.755 |      0.910      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth) |
| [SimpleBaseline-ResNet152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192.py) |  256x192   | 0.725 |      0.909      |      0.843      | 0.758 |      0.913      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth) |
| [HRNet+UDP-W32](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp.py) |  256x192   | 0.733 |      0.909      |      0.854      | 0.766 |      0.916      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth) |
| [HRNet+UDP-W48](configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_udp.py) |  256x192   | 0.752 |      0.910      |      0.855      | 0.784 |      0.918      | [ckpt](https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_256x192_udp-2554c524_20210223.pth) |


## Visualizing 2D Pose Results
- Please refer to the visualization script under ```./scripts/benchmarks/pose2d/1_vis_pose2d.sh```
- Modify the variables ```$CONFIG_FILE``` and ```$CHECKPOINT``` to select a detector. Refer to the [model zoo](https://github.com/rawalkhirodkar/egohumans/tree/main/egohumans/external/mmpose/configs) for supported methods.
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variable ```$SAVE_BIG_SEQUENCE_NAME``` to select the combination of big sequences used for evaluations.
- Modify the variable ```$MODE``` to be either ```ego_rgb```, ```ego_slam``` or ```exo```.
- Run the script
```shell
cd scripts/benchmarks/pose2d
chmod +x 1_vis_pose2d.sh
./1_vis_pose2d.sh
```
- The code will visualize the poses along with the ground truth under the folder ```$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/all/output/pose2d/$MODE```.
<span style="color:cyan;">Ground truth poses</span> are visualized in blue and <span style="color:orange;">predicted poses</span> are visualized in orange as shown.

<div style="text-align:center;">
    <img src="images/pose2d.gif" alt="Fencing 2d pose estimation">
</div>