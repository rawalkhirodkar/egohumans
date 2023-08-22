# Installation

EgoHumans works on Linux. It requires Python 3.8+, CUDA 11.2+ and PyTorch 1.8+.


## Step-by-Step Guide

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name eh python=3.10 -y
conda activate eh
```

**Step 2.** Install PyTorch with GPU following [official instructions](https://pytorch.org/get-started/locally/).

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**Step 3.** Install mmcv, mmdet, mmpose and mmhuman3d from source provided in this repository.

```shell
cd egohumans/external/mmcv
pip install -r requirements/build.txt
MMCV_WITH_OPS=1 pip install -v -e .  
```

```shell
cd ../mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

```shell
cd ../mmpose
pip install -r requirements.txt
pip install -v -e . 
```

```shell
cd ,,/mmhuman3d
pip install -v -e .
```

**Step 4.** Install other dependencies like pycococreator, pycolmap, etc. A fork of pycococreator is provided in egohumans/external for ease of use.
```shell
cd ../pycococreator
python setup.py install

pip install hdbscan yacs Rtree pyntcloud pyvista python-fcl pykalman torchgeometry colour pycolmap flask timm

```

## From scratch script

You can also execute the above steps using the provided shell script.

```shell
cd scripts/_install
chmod +x conda.sh

./conda.sh
```