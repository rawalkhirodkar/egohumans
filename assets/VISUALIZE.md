# Visualization



## 3D Poses

- Please use the provided visualization script for 3D poses under ```./scripts/vis/0_poses3d.sh```

- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variables ```$BIG_SEQUENCE``` and ```$SEQUENCE``` to the sequence and subsequence used for visualization.
- Run the script
```shell
cd scripts/vis
chmod +x 0_poses3d.sh
./0_poses3d.sh
```
- The code will project the 3D poses to multiple camera views (ego + exo) and save the images to ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/processed_data/vis_fit_poses3d```

## Meshes (SMPL)

- Please use the provided visualization script for 3D poses under ```./scripts/vis/1_smpl.sh```

- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variables ```$BIG_SEQUENCE``` and ```$SEQUENCE``` to the sequence and subsequence used for visualization.
- Run the script
```shell
cd scripts/vis
chmod +x 1_smpl.sh
./1_smpl.sh
```
- The code will project the 3D SMPL meshes to multiple camera views (ego + exo) and save the images to ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/processed_data/vis_smpl```

## Meshes (SMPL) using Blender

- First undistort all the camera images (ego and exo) using the script ```./scripts/data_process/0_undistort_images.sh```. [Reference to undistort](assets/DATA_PROCESS.md).
- Make sure you see the RGB images undistorted under the ego and exo folders ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/ego/$ARIA/images/undistorted_rgb``` and ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/exo/$CAMERA/undistorted_images``` respectively.
- Download and install [Blender](https://www.blender.org/download/). Make sure blender-python is installed and you can run blender using python scripts.
- It is important to enable GPU usage with blender while rendering to speed things up. Enable this inside Blender by going to EDIT -> Preferences -> Select GPUs.
- Please use the provided visualization script for 3D poses under ```./scripts/vis/2_smpl_blender.sh```
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variables ```$BIG_SEQUENCE``` and ```$SEQUENCE``` to the sequence and subsequence used for visualization.
- Run the script
```shell
cd scripts/vis
chmod +x 2_smpl_blender.sh
./2_smpl_blender.sh
```
- The code will project the 3D SMPL meshes using Blender to multiple camera views (ego + exo) and save the images to ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/processed_data/vis_smpl_cam_blender```.
- Feel free to utilize multiple gpus by distributing different time steps across gpus using the flags ```start_time``` and ```end_time```.
