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

---

## Predicting brain age with deep learning from raw imaging data results in a reliable and heritable biomarker
predict chronological age in healthy people. deviations from healthy brain aging have been associated with cognitive impairment and disease. CNN applied to both pre-processed and raw T1-weighted MRI data.

demonstrate the accuracy of CNN brain-predicted age using 2001 data of healthy adults.

heritability (遗传) of brain-predicted age using 62 female twins.

...

年龄预测与这次调研无关，略。

---

## Towards Alzheimer's Disease Classification through Transfer Learning

Marcia Hon, Naimul Mefraz Khan
Ryerson University, Toronto

**method**: **transfer learning**, where architecture such as **VGG** and **Inception** are initialized with **pre-trained** weights. fully-connected layer is re-trained with MRI images. **informative slices selection** based on **image entropy**. **OASIS** MRI **dataset**.

**pros**: training size is 10 times smaller than the state-of-the-art. comparable or better performance achieved.

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

​	1) **VGG-16** [17]

​	2) **Inception V4** [18]

B.most informative training data selection [19].

信息最多的slice具有最大的熵

calculate the *image entropy* of each slice. For a set of symbols with probabilities p1, p2, ..., pM, the entropy:
$$
H=-\sum_{i=1}^{M}{p_ilog p_i.}
$$

[19] C. Studholme, D. L. Hill, and D. J. Hawkes, “An overlap invariant
**entropy measure** of **3d** **medical** **image** alignment,” Pattern recognition,
vol. 32, no. 1, pp. 71–86, 1999.

**4.Experimental results**

experiments on two aforementioned CNN architectures.

Tools: Keras build [model](https://github.com/flyyufelix/cnn_finetune). MATLAB training data selection.

**A. dataset**

[OASIS](http://www.oasisbrains.org) [20] : 416 subject. randomly choose 200 subjects, among which are 100 AD and 100 HC. **CRD** variable ranging from 0 (HC) to 2 (greater than 0 are AD). 目前在通过ftp下载（用ubantu上的lftp，服务器信息见OASIS的*ftp instruction*）

![1545378596972]({{site.url}}/static/img/posts/1545378596972.png)

left: AD, right: HC

pick the most informative 32 images from the axial plane of each 3D scan. 6400 Training images, 3200 of which were AD and the other were HC.

image resized to 150x150 for VGG16, and 299x299 for Inception V4.

参照：[blog: ranked pre-trained models](http://localhost:4000/model/2018/12/21/pre-trained-model.html)

**B.Accuracy results**

5-fold cross-validation; VGG16, 100 epochs, batch size 40; inception V4, 100 epochs, batch size 8. 都是100 epochs，只是batch size不同。

optimizer: VGG16 RMSProp, inception V4 sgd lr=0.0001

[良心作者之代码和数据](https://github.com/
marciahon29/Ryerson MRP) 

accuracy: VGG16从零开始 (74.12%)、VGG16 transfer learning (92.3%)、Inception V4 transfer learning (96.25)

**C.Comparison with other methods**

![1545404515604]({{site.url}}/static/img/posts/1545404515604.png)

---

## Analysis of Alzheimer' Disease Based on the Random Neural Network Cluster in fMRI - 2018

Xia-an Bi, Qin Jiang et al.

College of information Science and Engineering, Hunan Normal University, 湖南师范...orz

问题：AD、HC二分类

思路：ensemble多个神经网络，发现还能feature selection找到疾病区域

**1.概要**

neuroimaging techniques such as EEG [Engels 2015], SPECT [Prosser 2015], PET [Pagani 2017], MEG [Engels 2016], **fMRI** [Graffanti 2015].

related works using deep learning: 

* classify MCI patients from HC [Suk 2016], 

* diagnose AD with 2D to 3D CNN [Gao 2017], 

* classify AD from HC [Ortiz 2016], [Ortiz 2013], [Luo 2017], [Suk 2014], 

本文还提出“高维特征在特征降维时将被丢失”的观点，未加解释，所以有待商榷。

本文两部分工作：

* AD和HC的二分类
* 特征选择

第一步：从BP, Elman NN, PNN, LVQ NN, Competitive NN发现**Elman NN**最适合特征选择；

第二步：随机Elman NN聚类用来选择特征，用于找出疾病区域。

最终找出23个疾病区域，分类精度92.31%。

**2.数据集和方法**

**subjects 受试者**

数据来自ADNI dataset，2个选择标准

* resting-state（ 人处于清醒放松状态下） fMRI data，
* mini-mental state examination (MMSE 评估认知障碍)  与 clinical dementia rating (CDR 临床痴呆评定) scores相吻合

筛选得到61 subjects in which are 25 AD and 36 HC.

**数据采集和预处理**

仪器采集的参数：canning images were acquired on a Philips Medical Systems 3 Tesla MRI scanner. Acquisition parameters included: pulse sequence = GR, TR = 3,000 ms, TE = 30 ms, matrix = 64∗64, slice thickness = 3.3 mm, slice number = 48, flip angle = 80◦.

预处理：降低信噪比，用DPARSF软件 [Chao-Gan 2010], [Wang 2013]。

“including: convert DICOM to NIFTI; removing first 10 time points; slicing timing; realigning [Jenkinson 2002]; normalizating images into the echo planar imaging (EPI) template (Misaki et al., 2010); temporal smoothing; removing the effect of low-level (<0.01 HZ) and high-level (>0.08 HZ) noise by a high-pass temporal filtering (Challis et al., 2015); removing covariates such as the whole brain signal and cerebrospinal fluid signal.”

**Functional Connectivity of Brain**

预处理后，分析FCB，选择FC为特征。

第一步：AAL brain atlas (Roll 2015)把脑分成90个区域;

第二步：提取各个区域的时序；

第三步：用Pearson 相关系数衡量区域两两之间的**FC（Friston 1993）**；

最终，每个受试者有4005 (90x89/2)个FC特征。