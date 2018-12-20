---
layout: post
title: Brain neural and deep learning
date:   2018-12-20 13:01:00 +0800
categories: [brain]
---

## Deep structural learning for classification of Alzheimer's disease

N. Oishi et al.

Kyoto University, Research and Educational Unit of Leaders for Integrated Medical System- Center for the Promotion of Interdisciplinary Education and Research, Kyoto, Japan

MRI, early detection of Alzheimer;s disease (AD). Deep learning overcome bias of manual feature extraction. 3D CNN for classifying AD patients and healthy controls (HCs).

Patients: 50 AD and 50 HCs. **3D T1-weighted structural MRI** scans in the **ADNI2**.

Material: structural MRIs were spatially normalized, **segmented** into **gray** matter, and smoothed with **[SPM8](http://www.neuro.uni-jena.de/vbm/download/)** and **[VBM8](http://www.neuro.uni-jena.de/vbm/download/)** Toolbox.

Methods: **4 convolutional layers, 2 fully connected layers, 5-fold cross-validations**.

Results: **accuracy 0.85**



## Predicting brain age with deep learning from raw imaging data results in a reliable and heritable biomarker
predict chronological age in healthy people. deviations from healthy brain aging have been associated with cognitive impairment and disease. CNN applied to both pre-processed and raw T1-weighted MRI data.

demonstrate the accuracy of CNN brain-predicted age using 2001 data of healthy adults.

heritability (遗传) of brain-predicted age using 62 female twins.

...

年龄预测与这次调研无关，略。

## Towards Alzheimer's Disease Classification through Transfer Learning

Marcia Hon, Naimul Mefraz Khan
Ryerson University, Toronto

---

**method**: **transfer learning**, where architecture such as **VGG** and **Inception** are initialized with **pre-trained** weights. fully-connected layer is re-trained with MRI images. **informative slices selection** based on **image entropy**. **OASIS** MRI **dataset**.

**pros**: training size is 10 times smaller than the state-of-the-art. comparable or better performance achieved.

---

**1.introduction**

every 85 people will be affected by AD by 2050 [1]. algorithm predicted better than clinicians [2]. example: SVM [3], then goes CNN.

current limitations [4], [5]: 1) require **large amount of labeled training data**; 2) training such large amount of data require **huge amount computational resources;** 3) overfitting/underfitting.

solution: transfer learning [6], models like ImageNet [7], examples: networks trained on natural images used with medical images[8].

our work: investigate **two** popular **CNN** **architectures** (ImageNet and Inception) into an AD diagnosis problem. intelligent **training selection** [19] and **transfer learning**.

**2.related work**

SVM [3]2010, feed-forward neural network [9]2017. 

deep learning: Auto-encoder (AE) [10], 3D AE [11]. Stacked AE [12]. CNN such as LeNet and Inception [13].

in-depth discussion from scratch vs fine-tuning on some medical applications [14].

[14] N. Tajbakhsh, J. Y. Shin, S. R. Gurudu, R. T. Hurst, C. B. Kendall, M. B.
Gotway, and J. Liang, “Convolutional neural networks for **medical image**
**analysis**: **Full training or fine tuning?**” IEEE transactions on medical
imaging, vol. 35, no. 5, pp. 1299–1312, 2016.

**3.methodology**

A.CNN and Transfer Learning

​	1) VGG-16 [17]

​	2) Inception V4 [18]

B.most informative training data selection [19].

calculate the *image entropy* of each slice. For a set of symbols with probabilities p1, p2, ..., pM, the entropy:

$$
H=-\sum_{i=1}^{M}{p_ilog p_i.}
$$

[19] C. Studholme, D. L. Hill, and D. J. Hawkes, “An overlap invariant
**entropy measure** of **3d** **medical** **image** alignment,” Pattern recognition,
vol. 32, no. 1, pp. 71–86, 1999.