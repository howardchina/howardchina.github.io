---
layout: post
title: What makes for effective detection proposals?
date:   2018-12-11 16:12:00 +0800
categories: [proposals]
---

### What makes for effective detection proposals?

### 什么造就了高效的detection proposals?

作者：Jan Hosang<sup>1</sup>, Rodrigo Benenson<sup>1</sup>, Piotr Dollar<sup>2</sup>和Bernt Schiele<sup>1</sup>

单位：１马克斯普朗克信息学研究所，Max Planck Institute for Informatics；２Facebook 人工智能研究，Facebook AI Research (FAIR)

**摘要**：眼下最好的目标检测器都用了detection proposals来引导目标搜索，因此避免了在图上滑窗搜索这种开销大的方式。尽管detection proposals流行且用途广泛，但仍不确定用它们检测目标时用了什么权衡策略。本文深入分析12个proposals方法以及4种baselines，考虑在PASCAL, ImageNet和MS COCO数据库上的proposal repeatability（产生proposal的稳定性）和金标准召回率，以及对DPM（2010,　discriminatively trained part-based models）, R-CNN和Fast R-CNN检测性能的影响。本文的分析说明了对目标检测而言提高proposal localisation accuracy和提高召回率一样重要。本文提出一种新颖的测量方法，平均召回率（AR），奖励高召回率和好的localisation，与detection的性能出奇地一致。本文的结论说明了现有方法的通常存在的强项和缺点，为选择和调试方法提供了见解和测量方法。

**关键词**：Computer Vision, objection detection, detection proposals.


1.**简介**

广为人知的经典方法“滑窗”（Sliding window）。滑窗分类器的个数和被测窗的数量呈线性关系，每个尺度的检测在每张图上要１w到10w个窗，多尺度的数量更上升一个数量级。现在的数据集[4]-[6]（PASCAL, ImageNet, MS COCO）还要预测目标的纵横比，将检测空间增加到了每张图100w-1000w个窗。[7]-[11]（[Regionlets for Generic object detection](https://blog.csdn.net/autocyz/article/details/44919725 "Regionlets for Generic object detection in ICCV 2013"), [Rich feature hierarchies for accurate object detection and semantic segmentation](# "R-CNN"), [multibox](#, "Scalable, high quality object detection"), [SPPNet](#, "Spatical pyramid pooling in deep convolutional networks for visual recognition"), [segmentation driven object detection with fisher vectors](#,"segmentation driven object detection with fisher vectors")）计算上易处理和高检测质量之间的紧张关系，有一种detection proposals的方法可以解决[12]-[15]（[What is an object? CVPR2010](), [Constrained parametric min-cuts for automatic object segmentation CVPR2010](), [Category independent object proposals ECCV2010](), [Segmentation as selective search for object recognition ICCV2011]()）。
