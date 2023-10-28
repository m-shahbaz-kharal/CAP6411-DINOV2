#!/bin/bash
# python eval_scripts/eval_classifier.py \
#     --model dinov2_vits14 \
#     --dataset ImageNet:split=VAL:root=dataset:extra=dataset

python eval_scripts/eval_classifier.py \
    --model dinov2_vitb14 \
    --dataset ImageNet:split=VAL:root=dataset:extra=dataset

python eval_scripts/eval_classifier.py \
    --model dinov2_vitl14 \
    --dataset ImageNet:split=VAL:root=dataset:extra=dataset

python eval_scripts/eval_classifier.py \
    --model dinov2_vitg14 \
    --dataset ImageNet:split=VAL:root=dataset:extra=dataset