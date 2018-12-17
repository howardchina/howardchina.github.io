---
layout: post
title: Computer aided endoscopy diagnosis
date:   2018-12-17 09:50:00 +0800
categories: [gastric endoscopy,computer aided diagnosis]
---

## Deep sparse feature selection for computer aided endoscopy diagnosis

作者：Yang Cong丛杨 <sup>a,b</sup>, Yunsheng Yang杨云生 <sup>d</sup> el at.
单位：a.机器人国家重点实验室-沈阳自动化所-中科院-中国, b.Department of Computer Science-University of Rochester-USA, c.中国解放军总医院

---


核心贡献: 
1. group sparsity based feature selection. 从特征集中选出最相关的特征，并为其设置权重;
2. 超像素分割的特征提取，去除图像质量低的区域;
3. 构建数据库10000张图，标定3800张pixel-level（I800 dataset）和frame-level（I3000 dataset）的金标准。

本文并没有区分病灶的类型，只是划出了病灶区域。

---

补充：找到了相关的中文[《基于稀疏表达的胃部疾病检测》](https://www.scholarmate.com/scmwebsns/archiveFiles/downLoadNoVer.action?fdesId=k%252By%252BHpfOQiV%252F7Xomi1RCC1EWlOrUeC6u)。背景：中文发于2013年，同一个实验室做的，同样的通讯和医生，本文是2015年收录的。

---

以下为原文译文部分：



#### 摘要

开发了一个**计算机辅助诊断算法**，应用于基于视觉的内窥镜检查，用来检测和分类不正常的地方。

第一步，图像超像素分割，提取颜色、纹理特征，得到特征向量；第二步，用group sparsity 设计特征选择模型——DSSVM，预评估了图像质量。

**优点**，第一步比传统patch的方法更灵活和精准；第二步还为提取的特征指定了权重，兼顾计算复杂性和准确性。

用1284个志愿者的3800张图像构建了一个**胃部内窥镜数据库**。

---

**1.introduction**

美国每年24000例胃癌、4百万例胃溃疡[1]。

重点在食道和胃的多种内窥镜下病灶，并用计算机辅助诊断食管病变和胃病变的异常。不是代替专家做最后的诊断，而是提醒和协助专家，减轻体力工作。低分辨率胶囊内窥镜[2-5]不能调节高度和位置，只能用在小肠和结肠，不适用于胃部。相比之下，高分辨率的传统内窥镜操作更灵活。

单一颜色和纹理特征见**图1(a)**。

启发式合并特征：**图1(b)**四种特征。

特征选择模型[6,7]**图1(c)**。[8]神经网络特征选择，检测幽门螺杆菌感染；[7]特征选择（sequential forward floating selection）、SVM，两阶段的pipline；[6]基于SVM的递归特征消除（SVM-REF）做特征选择，见图1(c)给更重要的特征维度赋予更高的权重，但这些不重要的特征也被提出来了，所以还是耗时。

**图1(d)**提取重要关联的特征，不提取无用特征，前三个特征被选中并赋权。DSSVM能同时选中特征单元和特征维度。

![1545017300061]({{site.url}}/static/img/posts/1545017300061.png)

此外考虑特征提取，有别于从patch[4,2,9]提取，很难确定用多宽的patch（小patch信息不足，大patch太多干扰），因此用一种更灵活的超像素的特征提取。

由于光照和人体内部结构对成像质量的影响，还评估并丢弃了一些成像质量差的区域。

核心贡献：1.group sparsity实现的DSSVM；2.超像素；3.构建3800张图的数据库。

章节编排：第2章相关工作；第3章方法总概；第4章图像表示；第5章DSSVM；第6章试验结果；第7章总结。

**2.相关工作**

[10,11]综述计算机辅助诊断内窥镜。

按设备分为两类：（1）主动型，传统内窥镜[7]、NBI[12]、zoom[13,14]、共聚焦激光内窥镜(confocal laser endomicroscopy)[15]，从口探入体内，检查食道（gullet）、胃、十二指肠（duodenum）；（2）被动型、非侵入，无线胶囊（Wireless Capsule）内窥镜[2-5]，广泛用于肠道，2fps传输内部图像。

按胃肠道中的区域分为：esophagus（食管）[16]、胃[17,7]、小肠[2-5]、结肠[9,18]。

按病灶分：出血[2]、癌[19,17]、乳糜泻（Celiac disease）、幽门螺杆菌（Helicobacter pylori）[7]、息肉（polyps）[20,14]和溃疡[4]、（食管）动力学（食管的运动功能）[21]、肿瘤[6,7]，巴雷斯特食道症（Barrett's esophagus，一种食道细胞病变）、克隆氏症（Crohn's disease，发炎性肠道疾病），以及单纯分为正常和非正常[22]。

其他：检测感兴趣帧[3]、无线胶囊内窥镜彩色视频分割[23]、自动摘要[24]、聚类[25]。

本文在传统胃镜下检测食道疾病和胃病的不正常的地方。

理论上，计算机辅助诊断有两个关键点：（1）图像表示，从内窥镜图像里提取特征单元，小波[14]、Gabor[12]、傅利叶[13]、LBP纹理[14]、

