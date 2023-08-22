
source ~/anaconda3/etc/profile.d/conda.sh

cd ../..


conda create -n cliff python=3.10 -y
conda activate cliff

pip install -r requirements.txt

##https://github.com/pytorch/pytorch/issues/45028
# conda install pytorch torchvision cudatoolkit=11 -c pytorch-nightly -y 
pip install chumpy

## torch geometry error
# https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported