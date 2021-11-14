# LOST 
Pytorch implementation of the unsupervised object discovery method **LOST**. More details can be found in the paper:

**Localizing Objects with Self-Supervised Transformers and no Labels**, BMVC 2021 [[arXiv](https://arxiv.org/abs/2109.14279)]  
by *Oriane Siméoni, Gilles Puy, Huy V. Vo, Simon Roburin, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, Renaud Marlet and Jean Ponce*

<div>
  <img width="25.3%" alt="LOST visualizations" src="examples/LOST_ex0.png">
  <img width="27.5%" alt="LOST visualizations" src="examples/LOST_ex1.png">
  <img width="27.5%" alt="LOST visualizations" src="examples/LOST_ex2.png">
</div>  


\
If you use the **LOST** code or framework in your research, please consider citing:


```
@inproceedings{LOST,
   title = {Localizing Objects with Self-Supervised Transformers and no Labels},
   author = {Oriane Sim\'eoni and Gilles Puy and Huy V. Vo and Simon Roburin and Spyros Gidaris and Andrei Bursuc and Patrick P\'erez and Renaud Marlet and Jean Ponce},
   journal = {Proceedings of the British Machine Vision Conference (BMVC)},
   month = {November},
   year = {2021}
}
```

## Installation
### Dependencies

This code was implemented with python 3.7, PyTorch 1.7.1 and CUDA 10.2. Please install [PyTorch](https://pytorch.org/). In order to install the additionnal dependencies, please launch the following command:

```
pip install -r requirements.txt
```

### Install DINO
This method is based on DINO [paper](https://arxiv.org/pdf/2104.14294.pdf). The framework can be installed using the following commands:
```
git clone https://github.com/facebookresearch/dino.git
cd dino; 
touch __init__.py
echo -e "import sys\nfrom os.path import dirname, join\nsys.path.insert(0, join(dirname(__file__), '.'))" >> __init__.py; cd ../;
```

The code was made using the commit ba9edd1 of DINO repo (please rebase if breakage).

## Apply LOST to one image
Following are scripts to apply LOST to an image defined via the `image_path` parameter and visualize the predictions (`pred`), the maps of the Figure 2 in the paper (`fms`) and the visulization of the seed expansion (`seed_expansion`). Box predictions are also stored in the output directory given by parameter `output_dir`.

```
python main_lost.py --image_path examples/VOC07_000236.jpg --visualize pred
python main_lost.py --image_path examples/VOC07_000236.jpg --visualize fms
python main_lost.py --image_path examples/VOC07_000236.jpg --visualize seed_expansion
```

## Launching LOST on datasets
Following are the different steps to reproduce the results of **LOST** presented in the paper. 

### PASCAL-VOC
Please download the PASCAL VOC07 and PASCAL VOC12 datasets ([link](http://host.robots.ox.ac.uk/pascal/VOC/)) and put the data in the folder `datasets`. There should be the two subfolders: `datasets/VOC2007` and `datasets/VOC2012`. In order to apply lost and compute corloc results (VOC07 61.9, VOC12 64.0), please launch:
```
python main_lost.py --dataset VOC07 --set trainval
python main_lost.py --dataset VOC12 --set trainval
```

### COCO
Please download the [COCO dataset](https://cocodataset.org/#home) and put the data in  `datasets/COCO`. Results are provided given the 2014 annotations following previous works. The following command line allows you to get results on the subset of 20k images of the COCO dataset (corloc 50.7), following previous litterature. To be noted that the 20k images are a subset of the `train` set.
```
python main_lost.py --dataset COCO20k --set train
```

## Different models
We have tested the method on different setups of the VIT model, corloc results are presented in the following table (more can be found in the paper). 

<table>
  <tr>
    <th>arch</th>
    <th>pre-training</th>
    <th colspan="3">dataset</th>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th>VOC07</th>
    <th>VOC12</th>
    <th>COCO20k</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>DINO</td>
    <td>61.9</td>
    <td>64.0</td>
    <td>50.7</td>
  <tr>
  <tr>
    <td>ViT-S/8</td>
    <td>DINO</td>
    <td>55.5</td>
    <td>57.0</td>
    <td>49.5</td>
  <tr>
  <tr>
    <td>ViT-B/16</td>
    <td>DINO</td>
    <td>60.1</td>
    <td>63.3</td>
    <td>50.0</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>DINO</td>
    <td>36.8</td>
    <td>42.7</td>
    <td>26.5</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>Imagenet</td>
    <td>33.5</td>
    <td>39.1</td>
    <td>25.5</td>
  <tr>
</table>

\
Previous results on the dataset `VOC07` can be obtained by launching: 
```
python main_lost.py --dataset VOC07 --set trainval #VIT-S/16
python main_lost.py --dataset VOC07 --set trainval --patch_size 8 #VIT-S/8
python main_lost.py --dataset VOC07 --set trainval --arch vit_base #VIT-B/16
python main_lost.py --dataset VOC07 --set trainval --arch resnet50 #Resnet50/DINO
python main_lost.py --dataset VOC07 --set trainval --arch resnet50_imagenet #Resnet50/imagenet
```

## Towards unsupervised object detection
In this work, we additionally use LOST predictions to train object detection models without any human supervision. We explore two scenarios: class-agnostic (CA) and (pseudo) class-aware training of object detectors (OD).


### Evaluating LOST+CA (corloc results)
The predictions of the class-agnostic Faster R-CNN model trained using LOST boxes as pseudo-gt are stored in the folder `data/CAD_predictions`. In order to launch the corloc evaluation, please launch the following scripts. It is to be noted that in this evaluation, only the box with the highest confidence score is considered per image. 

```
python main_corloc_evaluation.py --dataset VOC07 --set trainval --type_pred detectron --pred_file data/CAD_predictions/LOST_plus_CAD_VOC07.json
python main_corloc_evaluation.py --dataset VOC12 --set trainval --type_pred detectron --pred_file data/CAD_predictions/LOST_plus_CAD_VOC12.json
python main_corloc_evaluation.py --dataset COCO20k --set train --type_pred detectron --pred_file data/CAD_predictions/LOST_plus_CAD_COCO20k.json
```

The following table presents the obtained corloc results.

<table>
  <tr>
    <th>method</th>
    <th colspan="3">dataset</th>
  </tr>
  <tr>
    <th></th>
    <th>VOC07</th>
    <th>VOC12</th>
    <th>COCO20k</th>
  </tr>
  <tr>
    <td>LOST</td>
    <td>61.9</td>
    <td>64.0</td>
    <td>50.7</td>
  <tr>
  <tr>
    <td>LOST+CAD</td>
    <td>65.7</td>
    <td>70.4</td>
    <td>57.5</td>
  <tr>
</table>

### Training the models
We use the [detectron2](https://github.com/facebookresearch/detectron2) framework to train a Faster R-CNN model with LOST predictions as pseudo-gt. In order to reproduce our results, please install the framework. 

We use the `R50-C4` model of Detectron2 with ResNet50 pre-trained with DINO self-supervision [model](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth). 

Details: 
- mini-batches of size 16 across 8 GPUs using SyncBatchNorm 
- extra BatchNorm layer for the RoI head after conv5, i.e., `Res5ROIHeadsExtraNorm` layer in Detectron2
- frozen first two convolutional blocks of ResNet-50, i.e., `conv1` and `conv2` in Detectron2.
- learning rate is first warmed-up for 100 steps to 0.02 and then reduced by a factor of 10 after 18K and 22K training steps 
- we use in total 24K training steps for all the experiments, except when training class-agnostic detectors on
the pseudo-boxes of the VOC07 trainval set, in which case we use 10K steps. 

## License
LOST is released under the [Apache 2.0 license](./LICENSE).
