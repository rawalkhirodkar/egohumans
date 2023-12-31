ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html  

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN mkdir -p /mmpose/data
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

RUN conda install ffmpeg -y
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -c bottler nvidiacub -y

RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"

RUN pip install mmdet
RUN pip install "mmcls<0.18.0" "mmtrack<0.9.0,>=0.8.0"

# Install MMHuman3D
RUN conda clean --all
RUN git clone git@github.com:open-mmlab/mmhuman3d.git /mmhuman3d
WORKDIR /mmhuman3d
RUN mkdir -p /mmhuman3d/data
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -v -e .

RUN pip install xtcocotools==1.9
RUN pip install numpy==1.19.3
RUN pip install scipy==1.7
RUN pip install flask