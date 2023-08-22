DATA_ROOT='/media/rawalk/disk1/rawalk/vitpose/data'
OUTPUT_ROOT='/media/rawalk/disk1/rawalk/vitpose/Outputs'

### cd to root
cd ../..

ln -sfn $DATA_ROOT data
ln -sfn $OUTPUT_ROOT Outputs


MPII_DATA_ROOT='/mnt/nas/rawal/Desktop/vitpose/data/mpii'
ln -sfn $MPII_DATA_ROOT data/mpii

CROWDPOSE_DATA_ROOT='/mnt/nas/rawal/Desktop/vitpose/data/crowdpose'
ln -sfn $CROWDPOSE_DATA_ROOT data/crowdpose


AIC_DATA_ROOT='/mnt/nas/rawal/Desktop/vitpose/data/aic'
ln -sfn $AIC_DATA_ROOT data/aic