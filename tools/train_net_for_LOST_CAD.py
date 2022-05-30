#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2021 Valeo Comfort and Driving Assistance. All rights reserved.
# Adapted from detectron2.

import logging
import os
import copy
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads

#*******************************************************************************
#********************** REGISTERING THE NECESSARY DATASETS *********************
import json
import detectron2.data
def register_voc_in_coco_style(
    voc2007_trainval_json_path="./datasets/voc_objects_2007_trainval_CAD_coco_style.json",
    voc2007_test_json_path="./datasets/voc_objects_2007_test_CAD_coco_style.json",
    voc2012_trainval_json_path="./datasets/voc_objects_2012_test_CAD_coco_style.json"):

    dataset_suffix = "coco_style"
    voc2007_trainval_dataset_name = f"voc_2007_trainval_CAD_{dataset_suffix}"
    voc2007_test_dataset_name = f"voc_2007_test_CAD_{dataset_suffix}"
    voc2012_trainval_dataset_name = f"voc_2012_trainval_CAD_{dataset_suffix}"

    print(f"Registering the '{voc2007_trainval_dataset_name}' from the json file {voc2007_trainval_json_path}")
    def voc2007_trainval_dataset_function():
        with open(voc2007_trainval_json_path) as infile:
            json_data = json.load(infile)
        return json_data["dataset"]
    detectron2.data.DatasetCatalog.register(
        voc2007_trainval_dataset_name, voc2007_trainval_dataset_function)
    detectron2.data.MetadataCatalog.get(voc2007_trainval_dataset_name).thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get(voc2007_trainval_dataset_name).evaluator_type = "coco"
    detectron2.data.MetadataCatalog.get(voc2007_trainval_dataset_name).split = detectron2.data.MetadataCatalog.get("voc_2007_trainval").split
    detectron2.data.MetadataCatalog.get(voc2007_trainval_dataset_name).year = detectron2.data.MetadataCatalog.get("voc_2007_trainval").year
    detectron2.data.MetadataCatalog.get(voc2007_trainval_dataset_name).name = voc2007_trainval_dataset_name

    print(f"Registering the '{voc2007_test_dataset_name}' from the json file {voc2007_test_json_path}")
    def voc2007_test_dataset_function():
        with open(voc2007_test_json_path) as infile:
            json_data = json.load(infile)
        return json_data["dataset"]
    detectron2.data.DatasetCatalog.register(
        voc2007_test_dataset_name, voc2007_test_dataset_function)
    detectron2.data.MetadataCatalog.get(voc2007_test_dataset_name).thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get(voc2007_test_dataset_name).evaluator_type = "coco"
    detectron2.data.MetadataCatalog.get(voc2007_test_dataset_name).split = detectron2.data.MetadataCatalog.get("voc_2007_test").split
    detectron2.data.MetadataCatalog.get(voc2007_test_dataset_name).year = detectron2.data.MetadataCatalog.get("voc_2007_test").year
    detectron2.data.MetadataCatalog.get(voc2007_test_dataset_name).name = voc2007_test_dataset_name

    print(f"Registering the '{voc2012_trainval_dataset_name}' from the json file {voc2012_trainval_json_path}")
    def voc2012_trainval_dataset_function():
        with open(voc2012_trainval_json_path) as infile:
            json_data = json.load(infile)
        return json_data["dataset"]
    detectron2.data.DatasetCatalog.register(
        voc2012_trainval_dataset_name, voc2012_trainval_dataset_function)
    detectron2.data.MetadataCatalog.get(voc2012_trainval_dataset_name).thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get(voc2012_trainval_dataset_name).evaluator_type = "coco"
    detectron2.data.MetadataCatalog.get(voc2012_trainval_dataset_name).split = detectron2.data.MetadataCatalog.get("voc_2012_trainval").split
    detectron2.data.MetadataCatalog.get(voc2012_trainval_dataset_name).year = detectron2.data.MetadataCatalog.get("voc_2012_trainval").year
    detectron2.data.MetadataCatalog.get(voc2012_trainval_dataset_name).name = voc2012_trainval_dataset_name


def register_CAD_LOST_pseudo_boxes_for_the_voc2007_trainval_dataset(
    voc2007_json_path="./datasets/voc_2007_trainval_LOST_CAD.json",
    voc2007_dataset_name="voc_2007_trainval_LOST_CAD"):

    print(f"Registering the '{voc2007_dataset_name}' from the json file {voc2007_json_path}")
    def voc_2007_trainval_dataset_function():
        with open(voc2007_json_path) as infile:
            json_data = json.load(infile)
        return json_data["dataset"]
    detectron2.data.DatasetCatalog.register(
        voc2007_dataset_name, voc_2007_trainval_dataset_function)
    detectron2.data.MetadataCatalog.get(voc2007_dataset_name).thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get(voc2007_dataset_name).evaluator_type = "coco"

def register_CAD_objects_coco_train_dataset(image_root=None):
    print(f"Registering the 'coco_train_CAD' for class agnostic object detection.")
    def coco_train_ca_dataset_function():
        coco_data_gt = detectron2.data.DatasetCatalog.get("coco_2014_train")
        coco_data_gt = copy.deepcopy(coco_data_gt)
        # Make the ground bounding boxes class agnostic (i.e., give to all of
        # them the category id 0).
        for i in range(len(coco_data_gt)):
            if image_root is not None:
                coco_data_gt[i]["file_name"] = \
                    coco_data_gt[i]["file_name"].replace('datasets/coco', image_root)
            for j in range(len(coco_data_gt[i]["annotations"])):
                coco_data_gt[i]["annotations"][j]["category_id"] = 0
        return coco_data_gt
    detectron2.data.DatasetCatalog.register(
        "coco_train_CAD", coco_train_ca_dataset_function)
    detectron2.data.MetadataCatalog.get("coco_train_CAD").thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get("coco_train_CAD").evaluator_type = "coco"
    detectron2.data.MetadataCatalog.get("coco_train_CAD").name = "coco_train_CAD"

def register_CAD_objects_coco_val_dataset(image_root=None):
    print(f"Registering the 'coco_val_CAD' for class agnostic object detection.")
    def coco_val_ca_dataset_function():
        coco_data_gt = detectron2.data.DatasetCatalog.get("coco_2014_val")
        coco_data_gt = copy.deepcopy(coco_data_gt)
        # Make the ground bounding boxes class agnostic (i.e., give to all of
        # them the category id 0).
        for i in range(len(coco_data_gt)):
            if image_root is not None:
                coco_data_gt[i]["file_name"] = \
                    coco_data_gt[i]["file_name"].replace('datasets/coco', image_root)
            for j in range(len(coco_data_gt[i]["annotations"])):
                coco_data_gt[i]["annotations"][j]["category_id"] = 0
        return coco_data_gt
    detectron2.data.DatasetCatalog.register(
        "coco_val_CAD", coco_val_ca_dataset_function)
    detectron2.data.MetadataCatalog.get("coco_val_CAD").thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get("coco_val_CAD").evaluator_type = "coco"
    detectron2.data.MetadataCatalog.get("coco_val_CAD").name = "coco_val_CAD"

def register_CAD_coco20k_train_gt_dataset(
    coco_json_path="./datasets/coco20k_trainval_CAD_gt.json",
    coco_dataset_name="coco20k_train_CAD_gt"):

    print(f"Registering the '{coco_dataset_name}' from the json file {coco_json_path}")
    def coco_train_dataset_function():
        with open(coco_json_path) as infile:
            json_data = json.load(infile)
        return json_data["dataset"]
    detectron2.data.DatasetCatalog.register(
        coco_dataset_name, coco_train_dataset_function)
    detectron2.data.MetadataCatalog.get(coco_dataset_name).thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get(coco_dataset_name).evaluator_type = "coco"

def register_CAD_LOST_pseudo_boxes_for_the_coco20k_trainval_dataset(
    coco20k_json_path="./datasets/coco20k_train_LOST_CAD.json",
    coco20k_dataset_name="coco20k_train_LOST_CAD"):

    print(f"Registering the '{coco20k_dataset_name}' from the json file {coco20k_json_path}")
    def coco20k_train_dataset_function():
        with open(coco20k_json_path) as infile:
            json_data = json.load(infile)
        return json_data["dataset"]
    detectron2.data.DatasetCatalog.register(
        coco20k_dataset_name, coco20k_train_dataset_function)
    detectron2.data.MetadataCatalog.get(coco20k_dataset_name).thing_classes = ["object",]
    detectron2.data.MetadataCatalog.get(coco20k_dataset_name).evaluator_type = "coco"


#*******************************************************************************
#*******************************************************************************
# Comment out those not needed.
# Register VOC datasets
register_voc_in_coco_style()
register_CAD_LOST_pseudo_boxes_for_the_voc2007_trainval_dataset()

# Register COCO dataset
register_CAD_coco20k_train_gt_dataset()
register_CAD_objects_coco_train_dataset(image_root='../datasets/COCO/images')
register_CAD_objects_coco_val_dataset(image_root='../datasets/COCO/images')
try:
    register_CAD_LOST_pseudo_boxes_for_the_coco20k_trainval_dataset()
except:
    print("If failing here, please make sure to construct pseudo-boxes dataset using:\
          >python tools/prepare_coco_LOST_CAD_pseudo_boxes_in_detectron2_format.py --pboxes /path/preds.pkl")
#*******************************************************************************
#*******************************************************************************

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
