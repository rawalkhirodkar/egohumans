# Data Download

EgoHumans data is hosted at [Google Drive](https://drive.google.com/drive/u/2/folders/1JD963urzuzV_R_6FOVOtlx8UupwUuknR). 

The entire dataset is about 550 GB. However, you can get started by downloading a subset of sequences as well.

## Directory Structure

- Create a root directory "data".

```shell
mkdir data
```

- Create directory for a sequence (say 01_tagging).
``` shell
cd data
mkdir 01_tagging
```

- Download the .tar.gz file for any of the subsequence (say 01_tagging/004_tagging) and place it under the sequence folder. 

- Untar the subsequence. Please use ```--strip-components=8``` to avoid generating nested empty parent folders.
``` shell
cd 01_tagging
tar -xzvf 004_tagging.tar.gz --strip-components=8
```
- Download the COCO style annotation files for the EgoHumans test set from [benchmark](https://drive.google.com/drive/folders/1cz1P-fO5bkZSbGhCqE1sABDIHHA74JWz?). 
Place the folder under the ```./data``` directory. We use the sequences ```01_tagging```, ```02_legoassemble``` and ```03_fencing``` as our test set.

- You data directory structure should look like.

```
${ROOT}
|-- data
    01_tagging
    |-- 001_tagging
        |-- colmap
        |-- ego
            |-- aria01
                |-- calib
                |-- images
                    |-- left
                    |-- rgb
                    |-- right
            |-- aria02
            |-- aria03
            |-- aria04
        |-- exo
            |-- cam01
                |-- images
            |-- cam02
            ..
            |-- cam15
        |-- processed_data
            |-- fit_poses3d 
            |-- smpl 
            |-- init_smpl
            |-- bboxes
            |-- poses2d
            |-- poses3d
            |-- refine_poses3d
    benchmark
    |-- 01_tagging:02_legoassemble:03_fencing
        |-- all
            |-- coco
                |-- person_keypoints_ego_rgb.json
                |-- person_keypoints_ego_slam.json
                |-- person_keypoints_exo.json
            |-- coco_track
                |-- tracking_ego_rgb.json
                |-- tracking_ego_slam.json
                |-- tracking_exo.json
                |-- detections_ego_rgb.pkl
                |-- detections_ego_slam.pkl
                |-- detections_exo.pkl

```