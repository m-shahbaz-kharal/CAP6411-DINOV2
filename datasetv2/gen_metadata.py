from dinov2.data.datasets import ImageNet

# first set the names right
# import os
# src = os.listdir("datasetv2/val")
# src.sort(key=lambda x: int(x))
# trg = os.listdir("dataset/val")
# trg.sort(key=lambda x: int(x[1:]))

# for s,t in zip(src, trg): os.rename(f"datasetv2/val/{s}", f"datasetv2/val/{t}")

# i = 0
# v = 'ILSVRC2012_val_'
# for p in trg:
#     for f in os.listdir('datasetv2/val/'+p):
#         os.rename(os.path.join('datasetv2/val/'+p,f), os.path.join('datasetv2/val/'+p,v+str(i).zfill(8)+'.JPEG'))
#         i += 1

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="datasetv2", extra="datasetv2")
    dataset.dump_extra()