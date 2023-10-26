source ~/anaconda3/etc/profile.d/conda.sh

## cd to root of the repository
cd ../..

conda create -n midas python=3.8 -y
conda activate midas

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

pip install timm opencv-python