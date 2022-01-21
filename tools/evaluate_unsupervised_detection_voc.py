import argparse

import io
import json

import numpy as np
import os
import os.path
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache

import detectron2.data
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager

from scipy.optimize import linear_sum_assignment
from detectron2.structures import Boxes, BoxMode


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool_)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    if isinstance(detpath, dict):
        image_ids = detpath["image_ids"]
        confidence = detpath["confidence"]
        BB = detpath["BB"]
    else:
        detfile = detpath.format(classname)
        with open(detfile, "r") as f:
            lines = f.readlines()

        splitlines = [x.strip().split(" ") for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def hungarian_matching(reward_matrix):
    assert reward_matrix.shape[0] <= reward_matrix.shape[1], f"reward_matrix: {reward_matrix.shape}"
    class_ind, cluster_ind = linear_sum_assignment(-reward_matrix)
    map = reward_matrix[class_ind, cluster_ind].mean()

    cls_to_cluster = {cls: cluster for cls, cluster in zip(class_ind, cluster_ind)}

    if reward_matrix.shape[0] < reward_matrix.shape[1]:
        # Having more clusters than ground-truth classes.
        num_classes = reward_matrix.shape[0]
        num_clusters = reward_matrix.shape[1]
        cluster_to_cls = {cluster_ind[i]: class_ind[i] for i in range(num_classes)}
        cluster_ind_extra = list(set(range(num_clusters)).difference(set(cluster_ind)))
        #cluster_to_cls_extra = {c: num_classes + i for i, c in enumerate(cluster_ind_extra)}
        for i, c in enumerate(cluster_ind_extra):
            assert c not in cluster_to_cls
            cluster_to_cls[c] = num_classes + i
    else:
        cluster_to_cls = {cluster: cls for cls, cluster in zip(class_ind, cluster_ind)}

    return map, class_ind, cluster_ind, cls_to_cluster, cluster_to_cls


def load_predictions(results_file):
    with open(results_file) as infile:
        json_data = json.load(infile)

    predictions = defaultdict(list)
    detections = defaultdict(dict)
    for val in json_data:
        image_id = val["image_id"]
        category_id = val["category_id"]
        score = val["score"]
        bbox = val["bbox"]
        xmin, ymin, xmax, ymax = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        xmin += 1
        ymin += 1

        predictions[category_id].append(
            f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
        )

        if detections[category_id] == {}:
            detections[category_id] = {"image_ids": [], "confidence": [], "BB": []}
        detections[category_id]["image_ids"].append(image_id)
        detections[category_id]["confidence"].append(score)
        detections[category_id]["BB"].append([xmin, ymin, xmax, ymax])

    return predictions, detections


def sort_detections(detections):
    for cls_id in detections.keys():
        image_ids = detections[cls_id]["image_ids"]
        confidence = np.array(detections[cls_id]["confidence"])
        BB = np.array(detections[cls_id]["BB"]).reshape(-1, 4)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        detections[cls_id]["image_ids"] = image_ids
        detections[cls_id]["BB"] = BB
        detections[cls_id]["confidence"] = confidence[sorted_ind]

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='voc_2007_test')
    parser.add_argument('--results', type=str, default='./Pascal_Dino_ResNet50_faster_c4_voc07_based_on_lost_pseudo_boxes_clustered_with_k20/inference/coco_instances_results_voc_2007_test.json')
    args = parser.parse_args()

    meta = MetadataCatalog.get(args.dataset)
    # Too many tiny files, download all to local for speed.
    annotation_dir_local = PathManager.get_local_path(
        os.path.join(meta.dirname, "Annotations/"))
    args._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
    args._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
    args._class_names = meta.thing_classes

    predictions, detections = load_predictions(args.results)
    detections = sort_detections(detections)

    # Do hungarian matching between the clusters and the ground truth classes so
    # as to maximize the mean Average Precision (mAP).
    print("Hungarian matching...")
    num_classes = len(args._class_names)
    num_clusters = len(detections)
    reward_matrix = np.zeros([num_classes, num_clusters])
    with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
        for cls_id, cls_name in enumerate(args._class_names):
            for cluster_id in range(num_clusters):
                # Compute the AP for the class "cls_id" when using the
                # detections of the "cluster_id" cluster.
                _, _, reward_matrix[cls_id, cluster_id] = voc_eval(
                    detections[cluster_id], #res_file_template,
                    args._anno_file_template,
                    args._image_set_path,
                    cls_name, ovthresh=50/100.0, use_07_metric=False)
    map, _, _, cls_to_cluster, _ = hungarian_matching(reward_matrix)
    print(f"map: {map} at IoU 0.5")
    print(f"Class to cluster mapping: ==> {cls_to_cluster}")


    # Evaluate the detailed average precision results based on the cluster to
    # class mapping computed with hungarian_matching.
    with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
        res_file_template = os.path.join(dirname, "{}.txt")

        aps = defaultdict(list)  # iou -> ap per class
        for cls_id, cls_name in enumerate(args._class_names):
            for thresh in range(50, 100, 5):
                rec, prec, ap = voc_eval(
                    detections[cls_to_cluster[cls_id]], #res_file_template,
                    args._anno_file_template,
                    args._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=False,
                )
                aps[thresh].append(ap * 100)

    ret = OrderedDict()
    mAP = {iou: np.mean(x) for iou, x in aps.items()}
    ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
    for cls_id, cls_name in enumerate(args._class_names):
        apcoco = np.mean([aps[iou][cls_id] for iou in aps.keys()])
        print(f"{cls_name:20}: [AP50: {aps[50][cls_id]:10.3f} | AP: {apcoco:10.3f} | AP75: {aps[75][cls_id]:10.3f} ]")
    print("--------------")
    print(f'{"mean":20}: [AP50: {ret["bbox"]["AP50"]:10.3f} | AP: {ret["bbox"]["AP"]:10.3f} | AP75: {ret["bbox"]["AP75"]:10.3f} ]')
    print(ret["bbox"])
    print(f"{args.dataset}")
