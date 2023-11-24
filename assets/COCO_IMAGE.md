# Image based Benchmarks (COCO Format)

We provide annotation conversion scripts to convert our annotations into the more widely used COCO format.
This allows you to use our annotations with pre-existing methods and codebases for training and testing.

Additionally, you can skip this step by downloading the annotation files for our test set from [link](https://drive.google.com/drive/folders/1cz1P-fO5bkZSbGhCqE1sABDIHHA74JWz?usp=sharing).
Please download the ```benchmark``` folder and place under ```./data``` at the same level as other big sequence folders like ```01_tagging ... 07_tennis```.

## Converting a single sub-sequence annotations to the COCO Format

- Please use the provided annotation conversion script under ```./scripts/benchmarks/create_image_benchmarks/0_convert_to_coco_image_format.sh```
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variables ```$BIG_SEQUENCE``` and ```$SEQUENCE``` to the sequence and subsequence used for annotation conversion.
- Run the script
```shell
cd scripts/benchmarks/create_image_benchmark
chmod +x 0_convert_to_coco_image_format.sh
./0_convert_to_coco_image_format.sh
```
- The code will create three json files under the ```$SEQUENCE_ROOT_DIR/benchmark/$BIG_SEQUENCE/$SEQUENCE/coco``` directory.
    - person_keypoints_ego_rgb.json: 2D pose and bounding box annotations for RGB ego images.
    - person_keypoints_ego_slam.json: 2D pose and bounding box annotations for Grayscale (Left + Right) ego images.
    - person_keypoints_exo.json: 2D pose and bounding box annotations for RGB exo images.
    

## Concatenating multiple sub-sequence annotations into a single COCO file

- We combine annotations from multiple subsequences into single json files for ease of use.
- Please use the provided annotation concatenation script under ```./scripts/benchmarks/create_image_benchmarks/1_concatenate_coco_image_format.sh```
- Modify the variable ```$SEQUENCE_ROOT_DIR``` to point to the absolute path to the data.
- Modify the variable ```SAVE_BIG_SEQUENCE_NAME``` to be a ```:``` separated list of all the big sequences you wish to concatenate.
- Modify the variable ```BIG_SEQUENCES``` to be a ```:``` separated list of the big sequence name, repeat depending on how many subsequences it contains.
- Modify the variable ```SEQUENCES``` to be a ```:``` separated list of the subsequence names.
- Run the script
```shell
cd scripts/benchmarks/create_image_benchmark
chmod +x 1_concatenate_coco_image_format.sh
./1_concatenate_coco_image_format.sh
```
- The code will create three json files under the ```$SEQUENCE_ROOT_DIR/benchmark/$SAVE_BIG_SEQUENCE_NAME/all/coco``` directory.
    - person_keypoints_ego_rgb.json: 2D pose and bounding box annotations for RGB ego images.
    - person_keypoints_ego_slam.json: 2D pose and bounding box annotations for Grayscale (Left + Right) ego images.
    - person_keypoints_exo.json: 2D pose and bounding box annotations for RGB exo images.