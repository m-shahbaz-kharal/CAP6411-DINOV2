#!/bin/bash
python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vits14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vits14_pretrain.pth \
    --output-dir outputs/knn/l/ \
    --train-dataset ImageNet:split=TRAIN:root=dataset:extra=dataset \
    --val-dataset ImageNet:split=VAL:root=dataset:extra=dataset

python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitb14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vitb14_pretrain.pth \
    --output-dir outputs/knn/b/ \
    --train-dataset ImageNet:split=TRAIN:root=dataset:extra=dataset \
    --val-dataset ImageNet:split=VAL:root=dataset:extra=dataset

python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitl14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vitl14_pretrain.pth \
    --output-dir outputs/knn/large/ \
    --train-dataset ImageNet:split=TRAIN:root=dataset:extra=dataset \
    --val-dataset ImageNet:split=VAL:root=dataset:extra=dataset

python dinov2/run/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights pretrained/dinov2_vitg14_pretrain.pth \
    --output-dir outputs/knn/g/ \
    --train-dataset ImageNet:split=TRAIN:root=dataset:extra=dataset \
    --val-dataset ImageNet:split=VAL:root=dataset:extra=dataset