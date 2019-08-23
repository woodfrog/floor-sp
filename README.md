# Floor-SP: Inverse CAD for Floorplans by Sequential Room-wise Shortest Path

## Contents
1. [Overview](#overview)
2. [Data preprocessing](#data-preprocessing)
3. [Floor-SP](#Floor-SP)
	- [Mask-RCNN](#1-mask-rcnn)
	- [corner/edgeness modules](#2-corneredgeness-modules)
	- [Sequential room-wise shortest path ](#3-sequential-room-wise-shortest-path)
	- [Room merging ](#4-room-merging)
4. [Environment Setup](#environment-setup)
    - [Algorithm dependencies](#algorithm-dependencies)
5. [References](#references)


## Overview

This is the official implementation of the paper [Floor-SP: Inverse CAD for Floorplans by Sequential Room-wise Shortest Path](https://arxiv.org/abs/1908.06702), published on ICCV 2019.  

Floor-SP takes aligned RGBD scans of an indoor space as the input and produces floorplan estimation. The overall pipeline consists of 1) Data pre-processing and 2) Running Floor-SP. Detailed steps will be explained in following sections.

If you find the paper and the code helpful, please consider citing our paper:

```
@InProceedings{cjc2019floorsp, 
     title={Floor-SP: Inverse CAD for Floorplans by Sequential Room-wise Shortest Path}, 
     author={Jiacheng Chen, Chen Liu, Jiaye Wu, Yasutaka Furukawa}, 
     booktitle={The IEEE International Conference on Computer Vision (ICCV)}, 
     year={2019}
}

```

The data will be release by Beike(www.ke.com), the link will be provided soon.


## Data preprocessing

We need to pre-process aligned RGBD scans to get the global point cloud for an indoor space. The global point cloud will then be converted into top-view 2D density map and mean surface normal map, which are the inputs to our Floor-SP. In this part, we also parse the annotation files and roughly clean the annotations by filtering invalid geometric structures. 

1. Run `./utils/data\_process.py` to pre-process data. This merges local point clouds into a global one and parses annotations. This part also makes some simple cleaning on the annotations. Sometimes the given data does not contain the json annotation file, those samples are skipped. 
    
2. Run the data writer ./utils/data\_writer.py to generate training/testing data for Floor-SP. Floor-SP uses the files under `separate_room_data` directory. This writer is adapted from the data writer of FloorNet[2].

3. Note that the above instructions are for processing data with annotation files. **If the aim is to run pre-trained Floor-SP on new point clouds, we don't need annotations**. A simpler data pre-processing writer could be implemented here.


## Floor-SP

### 1. Mask-RCNN 

(All files are in `./mask-rcnn/`)
    
Mask-RCNN [3] is the first part in Floor-SP's pipeline. It generates room instance segmentations, these segmentations form the region term in our definition of the **room-aware floorplan reconstruction**. Also, the room masks also serve as inputs to modules predicting corner/edge likelihood maps.   

The implementation of this part is based on a public pytorch implementation of Mask-RCNN. See `./mask-rcnn/README.md` for information related to installation and environment setup. This mask-rcnn focuses on detecting room instances with segmentations. The data is by default stored in `./FloorNet/data/separate_room_data` after running the pre-processing.

`./mask-rcnn/main.py` is the script for training and evaluation of the trained mask-rcnn. We provide the pre-trained weights so ideally you don't need to re-train the model. We will only need to run the inference part for getting room segmentation results on new data.

The link to the pre-trained weights is [link to the pre-trained weights](https://drive.google.com/open?id=1Rb37cQd4gey2gYWvKL94VYBuRPFqJRVX). You can also find the weights for other modules of Floor-SP in the tar file.


- `./mask-rcnn/inference_corner.py` runs pre-trained mask-rcnn for room instance segmentation and prepares the data for **corner/edge estimation** for indoor scenes.


- `./mask-rcnn/inference_room.py` runs pre-trained mask-rcnn for room instance segmentation and prepares room-related data. The data can be used for training the room-corner association module, which predicts the affiliation between corner instance and room instance. This script also saves the labels/visualization colors of all room instances properly, which are important in generating visualization results.  
        

### 2. Corner/edgeness modules 

(see `./floor-sp/`)

Floor-SP uses Dialated ResNet (DRN) based neural networks [4] to estimate corner likelihood map and edgeness likelihood map. These maps are then converted into various energy terms in the **room-aware floorplan reconstruction** formulation.

The model is in `./floor-sp/models/corner_net`, `CornerEdgeNet` predicts corner/edgeness maps simultaneously.

`./floor-sp/mains/corner_main.py` is the script for the training / inference of Floor-SP's  corner+edgeness module. The module takes density map + mean surface normal map as input and generates corner / edgeness likelihood maps. These maps are important components of Floor-SP's overall formulation. 

Similarly, `./floor-sp/mains/associate_main.py` is the script for the training/ inference of a room-corner affiliation module. This module predicts which corner belongs to which room and is useful for reducing the search space for **room-wise coordinate descent**.  However, this is an optional module for Floor-SP since we found that simply using heuristics to pick up candidate corners for every room suffice. We provide the heuristics-based version in `./floor-sp/mains/associate_heuristics.py`. Running either script generates intermediate data for later steps. We use the DNN-based association module to generate results presented in the paper, but the heuristics-based version produces almost the same results.   

Detailed instructions can be checked in `./floor-sp/README.md`.


### 3. Sequential room-wise shortest path 
We devise **the room-wise coordinate descent strategy (sequential room-wise shortest path)** to optimize room structures for the floorplan. (details are in `./floor-sp/utils/floorplan_utils/`)

**Room-wise coordinate descent** solves the **room-aware floorplan reconstruction**, an energy minimization problem, by using dynamic programming as the solver. The paper [Piecewise Planar and Compact Floorplan Reconstruction from Images](https://dl.acm.org/citation.cfm?id=2679884) [1] uses shortest path algorithm to find an optimal global floorplan (actually an outer-most boundary) for an indoor space. Following this idea, our room-wise coordinate descent runs shortest path algorithm iteratively to solve the optimal per-room structure. The rooms are processed sequentially and there could be multiple rounds of optimization just as in traditional coordinate descent. 

The energy minimization problem is established using room instance segmentations, room corner likelihood maps and room edge likelihood maps that we generated in previous parts of Floor-SP. **`./floor-sp/mains/extract_floorplan.py` is the script for running room-wise coordinate descent, together with room merging and the final visualization**. The algorithm related code can be found in `./floor-sp/utils/floorplan_utils`. 


### 4. Room merging 

Room merging to get final floorplan. + Visualization / testing, etc. (`./floor-sp/utils/floorplan_utils/merge.py`)

The final stage of Floor-SP is to merge the optimal per-room structure to get the final floorplan. This is not the technical core. **Since the energy formulation in room-aware floorplan reconstruction takes the consistency between neighbouring rooms into consideration, the room structures (i.e. room loops) produced by room-wise coordinate descent are usually in good status already**. The merging stage is just for making the final visualization looks beautiful, there will not be new rooms or new corners created. In the merging stage, edges that are almost co-linear and close to each other are merged to be on the same straight line, then corners closed to each other within a threshold are also merged into one. 

**Note:** The implementation of the merging of co-linear edges is not optimal,
we simply shift all edges in the set of co-linear edges to make them on the same line with the first one in
the set, while the ideal implementation should compute an average location and
move all edges to that place. This simpler implementation works well for non-product-level use cases.  


## Environment Setup

The implementation is based on Python3.5 and Pytorch0.4.0. The file [**requirements.txt**](https://github.com/woodfrog/Lianjia-inverse-cad/blob/master/requirements.txt) contains related packages and their corresponding versions in the environment for running the whole Floor-SP. You can do `pip install -r requirements.txt` to install all of them. 

**Notice**: We need Tensorflow to run the RecordWriter in ./FloorNet during data pre-processing, tensorflow-gpu==1.4.0 was used but any version >=1.4.0 should also work. A separate data writer for Floor-SP only could be implemented so that the data writer does not rely on Tensorflow. 

### Algorithm Dependencies

Floor-SP makes use of several algorithms/models to build up the whole
floorplan reconstruction system, they are listed as follows:

- Dynamic programming in a 2-D grid for solving an optimal structure [1], based
  on which we designd the room-wise coordinate descent algorithm.

- Mask-RCNN for getting room instance segmentations [3].

- Dialated Residual Networks for implementing the room/edgeness module [4].



## References

[1] R. Cabral and Y. Furukawa. Piecewise planar and compact floorplan reconstruction from images. In IEEE Conference on Computer Vision and Pattern Recognition(CVPR), pages 628–635. IEEE, 2014

[2] C. Liu, J. Wu, and Y. Furukawa. Floornet: A unified framework for floorplan reconstruction from 3d scans. In Proceedings of the European Conference on Computer Vision(ECCV), pages 201–217, 2018

[3] K. He, G. Gkioxari, P. Dollar, and R. B. Girshick. Mask r-cnn. 2017 IEEE International Conference on Computer Vision (ICCV), pages 2980–2988, 2017

[4] F. Yu, V. Koltun, and T. A. Funkhouser. Dilated residual networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 636–644, 2017




