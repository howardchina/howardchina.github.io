---
layout: post
title:  Automatic Brain Labeling via Multi-Atlas Guided FCN
date:   2019-01-06 22:50:00 +0800
categories: [brain]
---

## Automatic Brain Labeling via Multi-Atlas Guided Fully Convolutional Networks

The labels of target image are what we want. The atlas provides necessary informations of labels as references. 

During training, the K most similar patches for a training patch are selected and input into this deep training model, so model can learn the relevant features and it is like data augmentation.

During testing, we did the same thing without the loss function.

**MA-FCN** 

Keywords: Brain image labeling, multi-atlas, fully convolutional network, patch-based labeling

Longwei Fang 1-2, **Dinggang Shen** 3 et al.,

1. Institute of Automation, Chinese Academy of Sciences, Beijing, China,
2. Center for Excellence in Brain Science and Intelligence Technology, Chinese Academy of Sciences, Beijing, China
3. Department of Radiology and BRIC, University of North Carolina at Chapel Hill, North Carolina, USA

**Abstract**:

traditional methods:

1. register multiple atlases to the target image (non-rigid alignment, lack high accuracy)
2. propagate the labels from the labeled atlases to the unlabeled target image

patch-based methods:

1. accurate registration (hand-crafted features)

deep learning:

* improve the labeling performance using the prior knowledge from the training atlases
* patch-based manner, where the input data consist of additional neighboring patches
* evaluated on several datasets

### 1.Introduction

Significance

* diagnosis
* investigating early brain development
* a fundamental step in brain network analysis pipeline

Challenges

* complex brain structures
* ambiguous boundaries between neighboring regions
* large variation of the same brain structure across different subjects

![1546697855319]({{site.url}}/static/img/posts/1546697855319.png)

multi-atlas is a standard approaches; effective and robust.

two steps:

1. registering the atlas image to the target image
2. propagating the label map to the target image

two categories:

1. registration-based

   * highly rely on non-rigid registration

   * time-consuming (Iglesias and Sabuncu 2015)

2. patch-based

   * exploring several neighboring patches within a local search region (Khalifa, Soliman, 2016, Pereira, Pinto, 2016, Zhang, Wang, 2017)
   * similarity between patches
     * feature extraction based on anatomical structures (Tu and Bai 2010, Zhang, Wang 2016)
     * intensity distributions (Hao, Wang 2014, Zikic, Glocker 2014)

   * lack of learned features

from ConvNet to FCN

In this paper, MA-FCN proposed for combining the good points of patch-based manner and registration-based.

* replace non-rigid registration to affine registration with 12 degree freedom
* a candidate target patch selection strategy is used for balancing the large variability of ROI sizes.
* input both target patches and their corresponding candidate atlas patches for training.
* atlas-unique pathway, target-patch pathway, and atlas-aware fusion pathway.

Contribution:

* multi-atlas training
* no need for non-rigid registration

### 2.Related Works

**Registration-based labeling**

* non-linear registration
* label fusion
* IT TAKES LOTS OF TIME TO ALIGN ATLAS TO THE TARGET IMAGE

**Patch-based labeling**

* non-local strategy
* propagate the label information of the selected similar atlas patches
* assuming only affine registration as a prerequisite

**ConvNet labeling**

* learn features

### 3.Method

*training* and *testing* stages

training

1. select **3D patches** from the training images randomly
2. for each selected 3D training patch, we select the ***K* most similar candidate atlas patches** within a specific search window.
3. training patches and their corresponding selected candidate atlas patches are input into the MA-FCN for training.
   * atlas patches are atlas intensity patches and label patches

testing

1. each testing 3D patch is concatented with its K most similar atlas patches, and the fed into MA-FCN to predict the patch
2. merge all the overlapping 3D patches by majority voting

***3.1 Data Preparation***

1. affine register all atlases to training data using FLIRT in FSL toolkit.
   * intensity images
   * and their corresponding label maps
2. patch sampling and selection strategy as shown in Figure 2

![1546754831689]({{site.url}}/static/img/posts/1546754831689.png)

***3.1.1 Training patch sampling***

boundary-focused patch extraction

* boundary patches : inside patches = 4:1

***3.1.2 Candidate atlas patch selection***

X: intensity image, Y: label map

1. normalization
2. select 3D candidates by 2-voxel stride step
3. select K 3D patches by one-voxel stride step from candidates

***3.2 Multi-atlas Guided Fully Convolutional Networks (MA-FCN)***

![1546758494158]({{site.url}}/static/img/posts/1546758494158.png)

***3.2.1 Atlas-unique pathway***

convert intensity and label into features

concatenate the atlas image and the target image together directly (Fang, Zhang et al. 2017)

* learn the mapping from intensity image to the label map
  * **Question: can this mapping be formulated in a connective way?**

in this study

* patch-wise 'atlas and target', an atlas-unique pathway

***3.2.2 Target-patch pathway***

U-Net

***3.2.3 Atlas-aware fusion pathway***

The feature maps in each level are concatenated together following several convolutions. Then a convolution layer with 1x1x1 kernel is used to fuse them together.

3.2.4 Loss function

cross-entropy loss

### 4.Experiments and Results

Evaluated the proposed method on

* [the LONI LCPA40 (Shattuck, Mirze et al. 2008) dataset](http://www.loni.ucla.edu/Atlases/LPBA40)
* [SATA MICCAI 2013 challenge dataset (Bennett Landman 2013)](https://masi.vuse.vanderbilt.edu/workshop2013/index.php/Main_Page)

which the two widely-used datasets for evaluating 2D or 3D labeling algorithms.

The LONI_LPBA40 dataset contains:

* 40 T1-weighted MR brain images
* with 54 manually labeled ROIs
* which mostly are distributed within cortical regions
* Here, the images and labels are used in this experiments

The SATA dataset has:

* 35 subjects with both intensity image and label map
* are provided with 14 manually labeled ROIs
* which are inner regions of the brain, covering accumbens, amygdala... on both hemispheres.
* both raw images and non-rigidly aligned images are provided by this dataset.

Implementation of this model

* CAFFE
* kernel weights initialize by Xavier function
* SGD for backpropagation
  * start learning rate = 0.01
  * inverse learning policy
    * gamma = 0.0001
    * momentum = 0.9
    * weight decay = 0.00005
* training batch size is 16 for LONI and 64 for SATA

index

* Dice Similarity Coefficient

* Hausdorff Distances

![1546778972062]({{site.url}}/static/img/posts/1546778972062.png)

***4.1 Evaluation on LONI LPBA40 dataset***

* 4-fold cross-validation

training

* training patch size 24x24x24

* 8100 patches from each image

* increase the number of data by densely cropping training patches from original MR image
* selection strategy
  * 150 from each ROI with
  * 120 from boundaries
  * and 30 from the inside of each ROI

testing

* fixed step size of 11 voxels
* majority voting for labeling overlapping patches

Selecting candidate atlas patches

* size of the search neighborhood = 12 voxels

number of candidate atlas patches is set to K=3

Figure 5 and 6. The labeling result is smoother than the ground truth.

![1546780225096]({{site.url}}/static/img/posts/1546778979999.png)

![1546780289571]({{site.url}}/static/img/posts/1546780289571.png)

***4.2 Evaluation on SATA MICCAI 2013 dataset***

7-fold cross-validation

* 2 folds as atlas images, 
* 4 folds as training set, 
* and the remaining one as test set

training

* training patch size = 12x12x12
* select 4200 patches from each training image.
* 300 patches are selected from each ROI
  * 240 around the boundary
  * 60 inside
* stride = 5 voxels
* search neighborhood = 12 voxels
* K = 3

***4.3 Parameter tuning***

***4.3.1 Patch size***

volumes of representative ROIs

* 12 ROIs from LONI_LPOBA40 dataset
* 6 ROIs from SATA MICCAI 2013 dataset

![1546784927345]({{site.url}}/static/img/posts/1546784927345.png)

![1546785002962]({{site.url}}/static/img/posts/1546785002962.png)

***4.3.2 The number of atlas-unique pathways***

![1546785920080]({{site.url}}/static/img/posts/1546785920080.png)

***4.4 Comparison with state-of-the-art methods***

The comparison methods include

1. HSPBL (Wu, Kim et al. 2015)
2. JLF (Wang, Suh et al. 2013) (antsJointFusion command in ANTs toolbox)

baseline: U-Net (target path), FCN (atlas path)

![1546786124735]({{site.url}}/static/img/posts/1546786124735.png)

![1546786159672]({{site.url}}/static/img/posts/1546786159672.png)