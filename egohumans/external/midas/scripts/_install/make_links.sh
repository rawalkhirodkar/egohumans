CHECKPOINTS_ROOT='/media/rawalk/disk1/rawalk/midas/weights'
INPUT_ROOT='/media/rawalk/disk1/rawalk/midas/input'
OUTPUT_ROOT='/media/rawalk/disk1/rawalk/midas/output'


### cd to root
cd ../..

ln -sfn $CHECKPOINTS_ROOT weights
ln -sfn $INPUT_ROOT input
ln -sfn $OUTPUT_ROOT output