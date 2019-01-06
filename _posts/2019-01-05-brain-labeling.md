---
layout: post
title:  Automatic Brain Labeling via Multi-Atlas Guided FCN
date:   2019-01-05 21:21:00 +0800
categories: [brain]
---

## Automatic Brain Labeling via Multi-Atlas Guided Fully Convolutional Networks

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

