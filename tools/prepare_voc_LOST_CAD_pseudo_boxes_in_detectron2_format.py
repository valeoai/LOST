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
import pdb
from os.path import join
from os import listdir, getcwd

import xml.etree.ElementTree as ET
import pathlib
import pickle
import json

import detectron2.data
from detectron2.structures import BoxMode


def get_img_size(ann_file):
    # Get the width and height from the annotation file.
    ann_file = open(ann_file)
    tree = ET.parse(ann_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    return width, height


def prepare_annotation_data(loc_object):
    if not isinstance(loc_object[0], (list, tuple)):
        loc_object = [loc_object,]

    annotations = []
    for obj in loc_object:
        xmin, ymin, xmax, ymax = [float(x) for x in obj]
        annotations.append({
            "iscrowd": 0,
            "bbox": [xmin, ymin, xmax, ymax],
            "category_id": 0,
            "bbox_mode": BoxMode.XYXY_ABS})

    return annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepares the LOST pseudo-boxes from a VOC"
                    "dataset in the data format expected from detectron2.")
    parser.add_argument("--voc_dir", type=str, default='../datasets/VOC',
                        help="Path to where the VOC dataset is.")
    parser.add_argument("--year", type=str, default='2007', help="Year of VOC dataset.")
    parser.add_argument("--pboxes", type=str, default='../outputs/VOC07_trainval/LOST-vit_small16_k/preds.pkl',
                        help="Path to where the LOST CA pseudo boxes for the VOCyear trainval data are.")
    args = parser.parse_args()

    # Dataset directory
    voc_dir = f"{args.voc_dir}{args.year}"

    # Load the boxes
    with open(args.pboxes, 'rb') as handle:
        LOST_pseudo_boxes = pickle.load(handle)

    data = []
    cnt = 0
    for image_name in LOST_pseudo_boxes:
        image_id = image_name[:-len('.jpg')]
        image_id_int = int(image_id)
        full_img_path = pathlib.Path(voc_dir) / "JPEGImages" / image_name
        full_ann_path = pathlib.Path(voc_dir) / "Annotations" / f"{image_id}.xml"
        width, height = get_img_size(full_ann_path)
        assert full_img_path.is_file()
        data.append({
            "file_name": str(full_img_path),
            "image_id": image_id,
            "height": height, "width": width,
            "annotations": prepare_annotation_data(LOST_pseudo_boxes[image_name]),
        })
        cnt += 1
    print(f'Number images saved {cnt}')
    dataset_name = f"voc_{args.year}_trainval_LOST_CAD"
    json_data = {
        "dataset": data,
        "meta_data": {
            "dirname": voc_dir,
            "evaluator_type": "pascal_voc",
            "name": dataset_name,
            "split": "trainval",
            "year": args.year,
            "thing_classes": "object",
        }}

    dst_file = f'./datasets/{dataset_name}.json'
    print(f"The pseudo-boxes at {args.pboxes} will be transformed into a detectron2-compatible dataset format at {dst_file}")
    with open(dst_file, 'w') as outfile:
        json.dump(json_data, outfile)