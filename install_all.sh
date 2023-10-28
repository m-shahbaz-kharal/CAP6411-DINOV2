#!/bin/bash
git clone https://github.com/facebookresearch/dinov2
cd dinov2
conda create -n dinov2 python=3.9
conda install -c "nvidia/label/cuda-11.7.1" cuda cuda-toolkit
pip install -r requirements.txt
pip install -r requirements-extras.txt
pip install -e .
