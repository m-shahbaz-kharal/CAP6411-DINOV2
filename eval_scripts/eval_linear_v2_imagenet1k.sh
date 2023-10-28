#!/bin/bash
python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vits14_pretrain.pth \
    --output-dir outputs/linear-v2/s/ \
    --train-dataset ImageNet:split=TRAIN:root=datasetv2:extra=datasetv2 \
    --val-dataset ImageNet:split=VAL:root=datasetv2:extra=datasetv2

python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitb14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vitb14_pretrain.pth \
    --output-dir outputs/linear-v2/b/ \
    --train-dataset ImageNet:split=TRAIN:root=datasetv2:extra=datasetv2 \
    --val-dataset ImageNet:split=VAL:root=datasetv2:extra=datasetv2

python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitl14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vitl14_pretrain.pth \
    --output-dir outputs/linear-v2/l/ \
    --train-dataset ImageNet:split=TRAIN:root=datasetv2:extra=datasetv2 \
    --val-dataset ImageNet:split=VAL:root=datasetv2:extra=datasetv2

python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vitg14_pretrain.pth \
    --output-dir outputs/linear-v2/g/ \
    --train-dataset ImageNet:split=TRAIN:root=datasetv2:extra=datasetv2 \
    --val-dataset ImageNet:split=VAL:root=datasetv2:extra=datasetv2