# Data Process

## How to undistort the images?

- Please use the undistortion script under ```./scripts/data_process/0_undistort_images.sh```

- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variables ```$BIG_SEQUENCE``` and ```$SEQUENCE``` to the sequence and subsequence used for visualization.
- Modify the variables ```$MODE``` to be either "all", "ego" or "exo" to undistort both ego and exo cameras, just ego cameras or just exo cameras
- Run the script
```shell
cd scripts/data_process
chmod +x 0_undistort_images.sh
./0_undistort_images.sh
```
- The code will undistort the images and save the ego images to ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/ego/$EGO_CAMERA/undistorted_rgb``` and exo images to ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/exo/$EXO_CAMERA/undistorted_images```.
