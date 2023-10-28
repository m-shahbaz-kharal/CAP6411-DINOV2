from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="dataset", extra="dataset")
    dataset.dump_extra()