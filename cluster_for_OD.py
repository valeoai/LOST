# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import pickle
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
import scipy.cluster.vq as vq

from networks import get_model
from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cluster LOST predictions.")
    
    # Model
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_small",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", 
        default=16, 
        type=int, 
        help="Patch resolution of the model."
    )
    
    # Dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    parser.add_argument(
        "--no_hard",
        action="store_true",
        help="Only used in the case of the VOC_all setup (see the paper).",
    )

    # Prediction files
    parser.add_argument(
        "--pred_file",
        type=str,
        default="outputs/VOC07_trainval/LOST-vit_small16_k/preds.pkl",
        help="Predicted boxes.",
    )

    # Clustering specific
    parser.add_argument(
        "--nb_clusters", 
        type=int, 
        default=20, 
        help="Number of clusters used for kmeans clustering.")

    parser.add_argument("--random_seed", 
        type=int, 
        default=123, 
        help="K-means random seed.")

    # Keep?
    parser.add_argument("--visualize", type=str, default=None, help="Visualize")


    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------------
    # Dataset
    dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, -1, device)

    # -------------------------------------------------------------------------------------------------------
    # Load predictions
    print(f'Extract features corresponding to the boxes {args.pred_file}.')

    with open(args.pred_file, "rb") as f:
        predictions = pickle.load(f)

    # -------------------------------------------------------------------------------------------------------
    # Extract CLS token

    # Features location
    out_path = f'{args.pred_file.split(".pkl")[0]}_cropped_feats_{args.arch}.pkl'

    if not os.path.exists(out_path):

        feats = defaultdict(defaultdict)

        pbar = tqdm(dataset.dataloader)
        for im_id, inp in enumerate(pbar):

            # ------------ Image processing ---------------------------------------
            img = inp[0]
            init_image_size = img.shape

            # Get the name of the image
            im_name = dataset.get_image_name(inp[1])

            # Pass in case of no gt boxes in the image
            if im_name is None:
                continue

            # Prediction
            pred = np.asarray(predictions[im_name])
            xmin, xmax = round(pred[1]), round(pred[3])
            ymin, ymax = round(pred[0]), round(pred[2])

            # Crop the image
            cropped = img[:, xmin:xmax, ymin:ymax]

            # Resize cropped region
            resize_f = pth_transforms.Resize(256, interpolation=3)
            cropped_im = resize_f(cropped)

            # move to gpu
            cropped_im = cropped_im.cuda(non_blocking=True)
            # Size for transformers
            w_featmap = cropped_im.shape[-2] // args.patch_size
            h_featmap = cropped_im.shape[-1] // args.patch_size

            # Forward pass
            with torch.no_grad():
                f = model(cropped_im[None, :, :, :])
                norm_f = nn.functional.normalize(f, dim=1, p=2)
                feats[im_name]["cropped_feat"] = np.array(norm_f.to("cpu"))
                feats[im_name]["predicted_bb"] = predictions[im_name]

        with open(out_path, "wb") as handle:
            pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'Cropped features saved at {out_path}.')

    else:
        with open(out_path, "rb") as f:
            feats = pickle.load(f)
        print(f'Cropped features loaded from {out_path}.')

    # -------------------------------------------------------------------------------------------------------
    # Apply clustering
    seed_ = f'_seed-{args.random_seed}' if args.random_seed != 123 else ""
    clustering_path = f'{args.pred_file.split(".pkl")[0]}_clustered_{args.nb_clusters}clu{seed_}.pkl'

    np.random.seed(seed=args.random_seed)
    all_feats = []
    pred_bbx = []

    keys = sorted(feats.keys())
    for key in keys:
        if feats[key]["cropped_feat"].squeeze().shape == (384,):
            all_feats.append(feats[key]["cropped_feat"].squeeze())
            pred_bbx.append(feats[key]["predicted_bb"])
            
    # Cluster whitened features
    x = np.array(all_feats)
    c, clusters = vq.kmeans2(data=vq.whiten(x) /  np.linalg.norm(vq.whiten(x), axis=1)[:, None], 
                             k=args.nb_clusters)

    pseudo_labels = defaultdict(defaultdict)
    for i in range(len(keys)):
        k = keys[i]
        pseudo_labels[k]["pseudo_label"] = clusters[i]
        pseudo_labels[k]["predicted_bb"] = pred_bbx[i]

    with open(clustering_path, "wb") as f:
        pickle.dump(pseudo_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Pseudo-labels saved at {clustering_path}.')