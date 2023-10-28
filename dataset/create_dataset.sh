#!/bin/bash
ln -s /datasets/ImageNet2012nonpub/train ./train
ln -s /datasets/ImageNet2012nonpub/validation ./val
ln -s /datasets/ImageNet2012nonpub/test ./test
wget https://github.com/facebookresearch/dinov2/files/11473738/labels.txt
conda activate dinov2
cd ..
python dataset/gen_metadata.py