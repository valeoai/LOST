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
from os.path import join

import xml.etree.ElementTree as ET
import pathlib
import json

from detectron2.structures import BoxMode


CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def get_img_size(ann_file):
    # Get the width and height from the annotation file.
    ann_file = open(ann_file)
    tree = ET.parse(ann_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    return width, height


def prepare_annotation_data(ann_file, class_agnostic=False):
    ann_file = open(ann_file)
    tree=ET.parse(ann_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    annotations = []
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)

        cls = obj.find('name').text
        if cls not in CLASSES or difficult==1:
            continue

        cls_id = 0 if class_agnostic else CLASSES.index(cls)

        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
        # Original annotations are integers in the range [1, W or H]
        # Assuming they mean 1-based pixel indices (inclusive),
        # a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        bbox[0] -= 1.0
        bbox[1] -= 1.0
        annotations.append({
            "iscrowd": 0, #difficult,
            "bbox": bbox,
            "category_id": cls_id,
            "bbox_mode": BoxMode.XYXY_ABS}) #
    return annotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc07_dir", type=str, default='../datasets/VOC2007',
                        help="Path where the VOC2007 data are.")
    parser.add_argument("--voc12_dir", type=str, default='../datasets/VOC2012',
                        help="Path where the VOC2012 data are.")
    parser.add_argument("--is_CAD", action='store_true', 
                        help="Are pseudo-boxes class-agnostic?")
    args = parser.parse_args()

    year2dir = {"2007": args.voc07_dir, "2012": args.voc12_dir}
    sets = [('2012', 'trainval'), ('2007', 'trainval'), ('2007', 'test'),]

    CAD_name = "_CAD" if args.is_CAD else ""

    for year, image_set in sets:
        image_ids = open(f'{year2dir[year]}/ImageSets/Main/{image_set}.txt').read().strip().split()
        print(f"==> Year: {year}, ImageSet: {image_set}, Number of images: {len(image_ids)}")
        data = []
        for image_id in image_ids:
            full_img_path = pathlib.Path(year2dir[year]) / "JPEGImages" / f"{image_id}.jpg"
            full_ann_path = pathlib.Path(year2dir[year]) / "Annotations" / f"{image_id}.xml"
            width, height = get_img_size(full_ann_path)
            assert full_img_path.is_file()
            data.append({
                "file_name": str(full_img_path),
                "image_id": image_id,
                "height": height, "width": width,
                "annotations": prepare_annotation_data(full_ann_path, args.is_CAD),
            })

        json_data = {
            "dataset": data,
            "meta_data": {
                "dirname": f"datasets/VOC{year}",
                "evaluator_type": "coco",
                "name": f"voc_{year}_trainval{CAD_name}_coco_style",
                "split": image_set,
                "year": int(year),
            }}

        dst_file = f'./datasets/voc_objects_{year}_{image_set}{CAD_name}_coco_style.json'
        print(f"Saving the coco-style voc data at {dst_file}")
        with open(dst_file, 'w') as outfile:
            json.dump(json_data, outfile)
