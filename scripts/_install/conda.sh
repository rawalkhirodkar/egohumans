#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

##-----------egohumans--------------------------
conda create -n eh python=3.10 -y
conda activate eh

## cd to egohumans repository
cd ../../egohumans

## install pytorch, make sure you have correct cuda version or feel free to change it
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

## install mmcv from source. make sure you have correct gcc version
cd external/mmcv
pip install -r requirements/build.txt
MMCV_WITH_OPS=1 pip install -v -e .  
echo -e "\e[32mMMCV installation success\e[0m"

## install mmdet
cd ../mmdetection
pip install -r requirements/build.txt
pip install -v -e .
echo -e "\e[32mMMDetection installation success\e[0m"

## install mmpose
cd ../mmpose
pip install -r requirements.txt
pip install -v -e . 
echo -e "\e[32mMMPose installation success\e[0m"

# ## install mmhuman3d
cd ../mmhuman3d
pip install -v -e .
echo -e "\e[32mMMHuman3D installation success\e[0m"

# ## install pycococreator
cd ../pycococreator
python setup.py install
cd ..

## install other dependencies
pip install hdbscan yacs Rtree pyntcloud pyvista python-fcl pykalman torchgeometry colour pycolmap flask timm

## install mmtracking
cd ../mmtracking
pip install -v -e .
pip install numpy --upgrade
cd ..
echo -e "\e[32mMMTracking installation success\e[0m"