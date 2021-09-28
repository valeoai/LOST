# LOST 
Pytorch implementation of the unsupervised object discovery method **LOST**. More details can be found in the paper:

**Localizing Objects with Self-Supervised Transformers and no Labels** [[arXiv](https://arxiv.org/abs/)]  
by *Oriane Siméoni, Gilles Puy, Huy V. Vo, Simon Roburin, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, Renaud Marlet and Jean Ponce*

<div>
  <img width="25%" alt="LOST visualizations" src="examples/LOST_ex0.png">
  <img width="31%" alt="LOST visualizations" src="examples/LOST_ex1.png">
</div>  


\
If you use the **LOST** code or framework in your research, please consider citing:


```
@article{LOST,
   title = {Localizing Objects with Self-Supervised Transformers and no Labels},
   author = {Oriane Sim\'eoni and Gilles Puy and Huy V. Vo and Simon Roburin and Spyros Gidaris and Andrei Bursuc and Patrick P\'erez and Renaud Marlet and Jean Ponce},
   journal = {arXiv preprint arXiv:},
   month = {09},
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

## Launching on datasets
Following are the command lines to reproduce results presented in the paper. 

### PASCAL-VOC
Please download the PASCAL VOC07 and PASCAL VOC12 datasets ([link](http://host.robots.ox.ac.uk/pascal/VOC/)) and put the data in the folder `datasets`. Their should be the two subfolders: `datasets/VOC2007` and `datasets/VOC2012`. In order to apply lost and compute corloc results (VOC07 61.9, VOC12 64.0), please launch:
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
