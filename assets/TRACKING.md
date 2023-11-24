# Tracking

We provide evaluation scripts to test offshelf tracking methods on the EgoHumans (ego-rgb) test set.


## Testing Trackers

- Please refer to the test script under ```./scripts/benchmarks/tracking/0_test_tracking.sh```
- Modify the variables ```$USE_GT_BBOX``` and ```$METHOD``` to select a detection mode and tracker for evaluation.
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variable ```$SAVE_BIG_SEQUENCE_NAME``` to select the combination of big sequences used for evaluations.
- Modify the variable ```$MODE``` to be either ```ego_rgb```, ```ego_slam``` or ```exo```. The code is tested with ```ego_rgb``` mode.
- Modify the variable ```$EVAL_TYPE``` to be either ```eval```, ```vis``` or ```debug``` for evaluation, qualitative visualization or interactive debugging.
- Run the script
```shell
cd scripts/benchmarks/tracking
chmod +x 0_test_tracking.sh
./0_test_tracking.sh
```
- The code will create a coco_track.pkl file under the folder ```$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/all/output/tracking/$METHOD/$MODE_$USE_GT_BBOX```.


## Results
We report the tracking results on the Ego-RGB images using YOLOX bounding box detections.
The metrics have a variance on 1% from the published results in Table. 1 of the paper due to pytorch upgrades, however the trends are similar.

|Method                                          | IDF1 |  MOTA   | MOTP | FP |  FN   | IDSw |                     
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | 
| Bytetrack |  50.9   | 59.7 |      78.9      |      20976      | 8124 |      2680      | 
| SimpleBaseline |  62.0   | 59.5 |      78.8      |      22901      | 7821 |      1195      |



