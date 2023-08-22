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
