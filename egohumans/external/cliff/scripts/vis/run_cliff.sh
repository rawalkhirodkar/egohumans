cd ../..

# ###-----------------------------------------------------------------------------
# # INPUT_VIDEO_FILE='/home/rawalk/Desktop/ego/cliff/data/test_samples/006_dodgeball_1/cam01/video.mp4'

# # INPUT_VIDEO_FILE='/home/rawalk/Desktop/ego/cliff/data/test_samples/006_dodgeball_1/aria01/rgb.mp4'
# # INPUT_VIDEO_FILE='/home/rawalk/Desktop/ego/cliff/data/test_samples/006_dodgeball_1/aria02/rgb.mp4'
# # INPUT_VIDEO_FILE='/home/rawalk/Desktop/ego/cliff/data/test_samples/006_dodgeball_1/aria03/rgb.mp4'
# INPUT_VIDEO_FILE='/home/rawalk/Desktop/ego/cliff/data/test_samples/006_dodgeball_1/aria04/rgb.mp4'


# ###-----------------------------------------------------------------------------
# CKPT_PATH=data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
# BACKBONE=hr48

# export CUDA_VISIBLE_DEVICES=0
# export EGL_DEVICE_ID=0
# python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
#                --input_path ${INPUT_VIDEO_FILE} --input_type video \
#                --show_bbox --show_sideView --save_results --make_video --frame_rate 20

###-----------------------------------------------------------------------------
# INPUT_FOLDER='/home/rawalk/Desktop/ego/cliff/data/test_samples/hugging/'
# INPUT_FOLDER='/home/rawalk/Downloads/temp/'
# INPUT_FOLDER='/home/rawalk/Desktop/ego/cliff/data/test_samples/hugging2'
# INPUT_FOLDER='/home/rawalk/Desktop/ego/cliff/data/test_samples/ballroom'
INPUT_FOLDER=/home/rawalk/Desktop/ego/cliff/data/test_samples/hi4d_pair00_dance

CKPT_PATH=data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
BACKBONE=hr48

export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0
python demo.py --ckpt ${CKPT_PATH} --backbone ${BACKBONE} \
                --input_path ${INPUT_FOLDER} --input_type folder \
                --show_bbox --show_sideView --save_results