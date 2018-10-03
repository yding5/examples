#!/bin/csh

#$ -M yding5@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -pe smp 2-4           # Specify parallel environment and legal core size
#$ -q gpu@qa-1080ti-008         # Specify queue (use ‘debug’ for development)
#$ -l gpu_card=1    # Specify number of GPU cards
#$ -N M00_4       # Specify job name

module load python/3.6.0 torch    # Required modules
#source /opt/crc/c/caffe/1.0.0-rc5/1.0.0
#python mydebug.py
setenv CUDA_VISIBLE_DEVICES 3
python3 neural_style/neural_style.py train --dataset ~/Dataset/COCO/train2014 --style-image images/style-images/mosaic.jpg --save-model-dir models/M00_4/ --epochs 2 --cuda 1
