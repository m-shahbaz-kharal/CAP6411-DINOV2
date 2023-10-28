import torch
import torch.nn as nn
from functools import partial

import numpy as np
import json

from dinov2.eval.linear import LinearClassifier, create_linear_input
from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.data.loaders import make_dataset
from dinov2.data.transforms import make_classification_eval_transform

import argparse

class Dino(nn.Module):
    def __init__(self, type="dinov2_vits14", pretrained=False):
        super().__init__()
        # get feature model
        model = torch.hub.load(
            "facebookresearch/dinov2", type, pretrained=pretrained
        ).cuda()
        autocast_ctx = partial(
            torch.cuda.amp.autocast, enabled=True, dtype=torch.float16
        )
        self.feature_model = ModelWithIntermediateLayers(
            model, n_last_blocks=1, autocast_ctx=autocast_ctx
        ).cuda()
        sample_input = torch.randn(1, 3, 224, 224).cuda()
        sample_output = self.feature_model(sample_input)

        # get linear readout
        out_dim = create_linear_input(
            sample_output, use_n_blocks=1, use_avgpool=True
        ).shape[1]
        self.classifier = LinearClassifier(
            out_dim, use_n_blocks=1, use_avgpool=True
        ).cuda()
        if pretrained:
            vits_linear = torch.load(f"pretrained/{type}_linear_head.pth")
            self.classifier.linear.load_state_dict(vits_linear)

    def forward(self, x):
        x = self.feature_model(x)
        x = self.classifier(x)
        return x
    
def get_files_names(dataset, index_start, index_end):
    entries = dataset._get_entries()
    actual_index = entries[index_start:index_end]["actual_index"]
    class_id = [dataset.get_class_id(index-1) for index in actual_index]
    image_relpath = [dataset.split.get_image_relpath(actual_index, class_id) for actual_index, class_id in zip(actual_index, class_id)]
    return image_relpath

def get_imagenet_real_accuracy(preds, val_fnames):
    with open('eval_scripts/real.json') as f: real_labels_json = json.load(f)
    predictions = np.argmax(preds, -1)

    val_fnames = [fname.split("/")[-1] for fname in val_fnames]

    # If the images were not sorted, then we need the filenames to map.
    real_labels = {f'ILSVRC2012_val_{(i+1):08d}.JPEG': labels for i, labels in enumerate(real_labels_json)}
    is_correct = [pred in real_labels[val_fnames[i]] for i, pred in enumerate(predictions) if real_labels_json[i]]
    real_accuracy = np.mean(is_correct)
    return real_accuracy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dinov2_vits14")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()

def main():
    args = get_args()
    model = Dino(type=args.model, pretrained=True)
    model.eval()

    dataset = make_dataset(dataset_str=args.dataset, transform=make_classification_eval_transform())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    file_names_list = []
    preds = []
    index_start = 0
    for i, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        labels = labels.cuda()
        index_end = index_start + len(images)
        file_names = get_files_names(dataset, index_start, index_end)
        outputs = model(images)
        
        file_names_list.extend(file_names)
        preds.append(outputs.detach().cpu().numpy())
        print(f"BATCH: {i+1} / {len(dataloader)}. Complete.")
        index_start = index_end
    
    preds = np.concatenate(preds)
    real_acc = get_imagenet_real_accuracy(preds, file_names_list) * 100.0
    # accuracy up to 2 decimal places
    print(f"Real Accuracy: {real_acc:.2f}")
    with open(f"{args.output_dir}/linear_real_acc_{args.model}.txt", "w") as f: f.write(f"{real_acc:.2f}")


if __name__ == "__main__": main()