# Video based Benchmarks (COCO Format)

We provide annotation conversion scripts to convert our video annotations into the more widely used COCO video format.
This allows you to use our annotations with pre-existing methods and codebases for training and testing.

Additionally, you can skip this step by downloading the annotation files for our test set from [link](https://drive.google.com/drive/folders/1cz1P-fO5bkZSbGhCqE1sABDIHHA74JWz?usp=sharing).
Please download the ```benchmark``` folder and place under ```./data``` at the same level as other big sequence folders like ```01_tagging ... 07_tennis```.

## Monocular Depth Estimation on Undistorted Ego-RGB Images using MiDas
- We use an offshelf monocular depth estimator [MiDas](https://github.com/isl-org/MiDaS) for 3D tracking using SimpleBaseline.
- Please install midas in a separate conda environment using the installation script under ```./egohumans/external/midas/scripts/_install/conda.sh```
- This will create a conda environment named ```midas```. Download the model weights for DPT-Large following the official instructions.
- Run the script under ```./scripts/benchmarks/create_video_benchmarks/0_get_depth.sh``` to run the depth estimator on undistorted ego-RGB Aria images.
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variables ```$BIG_SEQUENCE``` and ```$SEQUENCE``` to the sequence and subsequence used for annotation conversion.
- Run the script
```shell
cd scripts/benchmarks/create_video_benchmark
chmod +x 0_get_depth.sh
./0_get_depth.sh
```
- The code will save the MiDas depth estimates as .png and .pfm files under ```$SEQUENCE_ROOT_DIR/$BIG_SEQUENCE/$SEQUENCE/processed_data/depth``` directory.

## Converting a single sub-sequence annotations to the COCO Video Format
- Please use the provided annotation conversion script under ```./scripts/benchmarks/create_video_benchmarks/1_convert_to_coco_video_format.sh```
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variables ```$BIG_SEQUENCE``` and ```$SEQUENCE``` to the sequence and subsequence used for annotation conversion.
- Run the script
```shell
cd scripts/benchmarks/create_video_benchmark
chmod +x 1_convert_to_coco_video_format.sh
./1_convert_to_coco_video_format.sh
```
- The code will create three json and three pkl files under the ```$SEQUENCE_ROOT_DIR/benchmark/$BIG_SEQUENCE/$SEQUENCE/coco_track``` directory.
    - tracking_ego_rgb.json: Bounding box + 3D root location using MiDas + Person ID annotations for RGB ego images.
    - tracking_ego_slam.json.json: Bounding box + Person ID annotations for Grayscale (Left + Right) ego images.
    - tracking_exo.json: Bounding box + Person ID annotations for RGB exo images.
    - detections_ego_rgb.pkl: Helper information when evaluating 3D tracking using SimpleBaseline on RGB ego images.
    - detections_ego_slam.pkl: Helper information when evaluating 3D tracking using SimpleBaseline on Grayscale (Left + Right) ego images.
    - detections_exo.pkl: Helper information when evaluating 3D tracking using SimpleBaseline on RGB exo images.

## Concatenating multiple sub-sequence annotations into a single COCO Video file
- We combine video annotations from multiple subsequences into single json files for ease of use.
- Please use the provided video annotation concatenation script under ```./scripts/benchmarks/create_image_benchmarks/2_concatenate_coco_video_format.sh```
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variable ```SAVE_BIG_SEQUENCE_NAME``` to be a ```:``` separated list of all the big sequences you wish to concatenate.
- Modify the variable ```BIG_SEQUENCES``` to be a ```:``` separated list of the big sequence name, repeat depending on how many subsequences it contains.
- Modify the variable ```SEQUENCES``` to be a ```:``` separated list of the subsequence names.
- Run the script
```shell
cd scripts/benchmarks/create_video_benchmark
chmod +x 2_concatenate_coco_video_format.sh
./2_concatenate_coco_image_format.sh
```
- The code will create three json and three pkl files under the ```$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/all/coco_track``` directory.
    - tracking_ego_rgb.json: Bounding box + 3D root location using MiDas + Person ID annotations for RGB ego images.
    - tracking_ego_slam.json.json: Bounding box + Person ID annotations for Grayscale (Left + Right) ego images.
    - tracking_exo.json: Bounding box + Person ID annotations for RGB exo images.
    - detections_ego_rgb.pkl: Helper information when evaluating 3D tracking using SimpleBaseline on RGB ego images.
    - detections_ego_slam.pkl: Helper information when evaluating 3D tracking using SimpleBaseline on Grayscale (Left + Right) ego images.
    - detections_exo.pkl: Helper information when evaluating 3D tracking using SimpleBaseline on RGB exo images.