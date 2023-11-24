# Tracking

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
cd scripts/benchmarks/tracking
chmod +x 0_test_tracking.sh
./0_test_pose2d.sh
```
- The code will create a coco_bbox.pkl file under the folder ```$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/all/output/bbox/$MODE```.


## Results
We use ground-truth bounding boxes for evaluating the following methods. Click on the method name for the config file.

### Ego-RGB 
Total test images: 47740  
Total test instances: 78880




## Visualizing Tracking Results
