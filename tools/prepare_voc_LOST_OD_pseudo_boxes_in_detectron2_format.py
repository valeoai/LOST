# Copyright 2021 - Valeo Comfort and Driving Assistance
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

import argparse
import os
import xml.etree.ElementTree as ET
import pathlib
import pickle
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

import detectron2.data
from detectron2.structures import BoxMode

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor", None]


def get_img_size(ann_file):
    # Get the width and height from the annotation file.
    ann_file = open(ann_file)
    tree = ET.parse(ann_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    return width, height


def prepare_annotation_data(loc_object, cluster_to_cls):
    if not isinstance(loc_object, (list, tuple)):
        loc_object = [loc_object,]

    annotations = []
    for obj in loc_object:
        xmin, ymin, xmax, ymax = [float(x) for x in obj["predicted_bb"]]
        cluster_id = obj["pseudo_label"]
        if cluster_to_cls is None:
            category_id = cluster_id
        else:
            category_id = cluster_to_cls[cluster_id]
        annotations.append({
            "iscrowd": 0,
            "bbox": [xmin, ymin, xmax, ymax],
            "category_id": int(category_id),
            "bbox_mode": BoxMode.XYXY_ABS})

    return annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepares the clustered LOST pseudo-boxes from the VOC07 "
                    "dataset in the data format expected from detectron2.")
    parser.add_argument("--voc_dir", type=str, default='../datasets/VOC',
                        help="Path to where the VOC dataset is.")
    parser.add_argument("--year", type=str, default='2007', help="Year of VOC dataset.")
    parser.add_argument("--pboxes", type=str, default='',
                        help="Path to where the LOST clustered pseudo boxes for the VOC2007 trainval data are.")
    args = parser.parse_args()
    
    # Dataset directory
    voc_dir = f"{args.voc_dir}{args.year}"

    with open(args.pboxes, 'rb') as handle:
        LOST_pseudo_boxes = pickle.load(handle)

    cluster_ids = [v["pseudo_label"] for v in LOST_pseudo_boxes.values() if v != {}]
    num_clusters = max(cluster_ids) + 1
    cluster_to_cls = None

    data = []
    cnt = 0
    for file_name in LOST_pseudo_boxes.keys():
        image_id = file_name[:-len('.jpg')]
        image_id_int = int(image_id)
        full_img_path = pathlib.Path(voc_dir) / "JPEGImages" / file_name
        full_ann_path = pathlib.Path(voc_dir) / "Annotations" / f"{image_id}.xml"
        width, height = get_img_size(full_ann_path)
        assert full_img_path.is_file()
        data.append({
            "file_name": str(full_img_path),
            "image_id": image_id,
            "height": height, "width": width,
            "annotations": prepare_annotation_data(LOST_pseudo_boxes[file_name], cluster_to_cls),
        })
        cnt += 1
    print(f'Number images saved {cnt}')
    dataset_name = f"voc_2007_trainval_LOST_OD_clu{num_clusters}"
    json_data = {
        "dataset": data,
        "meta_data": {
            "dirname": voc_dir,
            "evaluator_type": "coco",
            "name": dataset_name,
            "split": "trainval",
            "year": 2007,
            "thing_classes": detectron2.data.MetadataCatalog.get(f"voc_2007_trainval").thing_classes,
        }}

    dst_file = f'./datasets/{dataset_name}.json'
    print(f"The pseudo-boxes at {args.pboxes} will be transformed into a detectron2-compatible dataset format at {dst_file}")
    with open(dst_file, 'w') as outfile:
        json.dump(json_data, outfile)
