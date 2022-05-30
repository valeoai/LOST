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
import cv2
import pdb
import matplotlib
import argparse
import datasets

import json
import torch
import torch.nn as nn
import torchvision
import numpy as np

from tqdm import tqdm

import pickle
from datasets import Dataset, bbox_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--type_pred",
        default="boxes_OD",
        choices=["boxes_OD", "detectron"],
        type=str,
        help="Type of predictions will inform on how to load",
    )
    parser.add_argument(
        "--pred_file", default="", type=str, help="File location of predictions."
    )
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

    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------------
    # Dataset
    dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Load predictions
    if not os.path.exists(args.pred_file):
        raise ValueError(f"File {args.pred_file} does not exists.")

    if args.type_pred == "boxes_OD":
        with open(args.pred_file, "rb") as f:
            predictions = pickle.load(f)
    elif args.type_pred == "detectron":
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)

    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])

        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)
        if gt_bbxs is not None:
            # Discard images with no gt annotations
            # Happens only in the case of VOC07 and VOC12
            if gt_bbxs.shape[0] == 0 and args.no_hard:
                continue

        if args.type_pred == "boxes_OD":
            pred = np.asarray(predictions[im_name])
        elif args.type_pred == "detectron":
            name_ind = im_name
            if "VOC" in args.dataset:
                name_ind = im_name[:-4]

            pred_ids = [
                id_i
                for id_i, pred in enumerate(predictions)
                if int(pred["image_id"]) == int(name_ind)
            ]

            # No predictions made
            if len(pred_ids) == 0:
                print("No prediction made")
                corloc[im_id] = 0
                cnt += 1
                continue

            # Select the most confident prediction
            confidence = [
                pred["score"]
                for id_i, pred in enumerate(predictions)
                if id_i in pred_ids
            ]
            most_confident = np.argsort(-np.asarray(confidence))[0]
            box = predictions[pred_ids[most_confident]]["bbox"]

            # From xywh to x1y1x2y2
            x1, x2 = box[0], box[0] + box[2]
            y1, y2 = box[1], box[1] + box[3]
            pred = np.asarray([x1, y1, x2, y2])

        ious = datasets.bbox_iou(
            torch.from_numpy(pred), torch.from_numpy(gt_bbxs.astype(np.float32))
        )

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
