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

（建议读原paper时自行脑补chinglish，减轻阅读障碍。）

**1. 概要**

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

**2. 数据集和方法**

**2.1 subjects 受试者**

数据来自ADNI dataset，2个选择标准

* resting-state（ 人处于清醒放松状态下） fMRI data，
* mini-mental state examination (MMSE 评估认知障碍)  与 clinical dementia rating (CDR 临床痴呆评定) scores相吻合

筛选得到61 subjects in which are 25 AD and 36 HC.

**2.2 数据采集和预处理**

仪器采集的参数：canning images were acquired on a Philips Medical Systems 3 Tesla MRI scanner. Acquisition parameters included: pulse sequence = GR, TR = 3,000 ms, TE = 30 ms, matrix = 64∗64, slice thickness = 3.3 mm, slice number = 48, flip angle = 80◦.

预处理：降低信噪比，用DPARSF软件 [Chao-Gan 2010], [Wang 2013]。

“including: convert DICOM to NIFTI; removing first 10 time points; slicing timing; realigning [Jenkinson 2002]; normalizating images into the echo planar imaging (EPI) template (Misaki et al., 2010); temporal smoothing; removing the effect of low-level (<0.01 HZ) and high-level (>0.08 HZ) noise by a high-pass temporal filtering (Challis et al., 2015); removing covariates such as the whole brain signal and cerebrospinal fluid signal.”

### **2.3 Functional Connectivity of Brain**（重要）

预处理后，分析FCB，选择FC为特征。

第一步：**AAL brain atlas (Roll 2015)**把脑分成90个区域;

第二步：提取各个区域的时序；

第三步：用Pearson 相关系数衡量区域两两之间的**FC（Friston 1993）**；

最终，每个受试者有4005 (90x89/2)个FC特征。

**2.4 随机神经网络聚类**

需要特征选择 [Azar 2015]，特征选择有不少方法也会丢失原始信息 [Zhou 2015, Jolliffe 2016, Alam 2017]。

本文提出一个随机神经网络聚类方法，通过随机选择样本和特征解决上述问题。（听起来要开始瞎掰了，带着些怀疑的态度看看他怎么解决的。）可以用于AD和HC分类，以及特征选择。该方法还能降维，同时避免步丢失重要信息，提高分类准确度。（接着看看怎么做到的）

**2.5 RNNC的设计和分类准确度**（有点儿不适了，设计是最开始的事情，分类准确度是最后的事情，并且我知道你肯定分类准确度提高了才会写papar，这么放一起合适嘛？）

思路：ensamble 多个CNN

第一步：数据集 D 分为训练结 N<sub>1</sub> 和测试集 N<sub>2 </sub>。HC为正类，AD为负类。

第二步：随机从训练集中选择出 n 个样本，随机从4005个特征中选择出 m 个特征。

第三步：然后构建一个神经网络，再回到第二步，重复执行k次，产生k个神经网络。

本方法和传统的方法的确不太一样。（更直观、更随机、更朴素）

接下来就是k个分类器投票，主要的结果就做为预测标签。

（所以... cluster不是聚类，是集群，clustering才是聚类，你做的是ensemble classifier吧!!!）

**2.6 从RNNC中提取特征**

反而言之，精度高的NN用的特征更好。统计出准确度高的哪些NN的特征频率，频率高的特征就是对于分类重要的。（回头看看abstract写的升华了不少，brain region把我唬住了

2.7 不正常的大脑区域

通过筛选出的特征，即连接两块区域的边，统计每个区域被连了多少次，找出不正常区域。

**2.8 实验设计**

第一步：训练分类器。构建4005个特征，划分训练集，训练 k=1000 个神经网络（NN）。

第二步：保留准确度大于0.6的 NN。

第三步：选择特征，保留240个特征。

第四步：确定最优特征个数。特征集有11种数量的特征，140, 150, 160, ..., 240。用这11种特征集构建KNNC，准确度最高的最优。

第五步：第一步到第四步，用BPNN、ELman NN、PNN、LVQ NN和Competitive NN都测一遍，准确度最高的最优。

第六步：通过第四步确定的特征确定不正常大脑区域。

**3. 结果**

3.1 受试人的年龄性别

t-test和chi-square test说明与AD和HC无关连。

3.2 分类结果

对随机Elman NN和随机PNN结果都还不错 Fig.3。180是最优特征数。（220也挺稳）从Fig. 3和Fig. 4看出random NN cluster比单一NN好。（哪儿看粗来了？从Fig. 3我只看出PNN和Elman NN都差不多，以及Fig. 4例PNN准确度比Elman高——慢着！Fig. 4画错了，Elman, LVQ和Competitive三者的bar怎么会完全一样？）准确度和运行时间见Table 2。

![Fig. 3]({{site.url}}/static/img/posts/1545642465704.png)

![1545642941231]({{site.url}}/static/img/posts/1545642941231.png)

3.3 不正常大脑区域

Tabel 3 这些区域的名字、缩写、体积。Figure 5 用Brain-NetView可视化的结果，Figure 6 是整体关系，Figure 7 是一对多的关系。

![Table 3]({{site.url}}/static/img/posts/1545647888906.png)

![Figure 5 6]({{url.site}}/static/img/posts/1545649887198.png)

![Figure 7]({{site.url}}/static/img/posts/1545650050718.png)

**DISCUSSION**

用特征提取所发现的PreCG和FG区域也被某些文章发现与AD有关。