---
layout: post
title:  Automatic Brain Labeling via Multi-Atlas Guided FCN
date:   2019-01-05 21:21:00 +0800
categories: [brain]
---

## Automatic Brain Labeling via Multi-Atlas Guided Fully Convolutional Networks

**MA-FCN**

Keywords: Brain image labeling, fully convolutional network, patch-based labeling

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

