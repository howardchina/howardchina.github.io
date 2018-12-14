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

广为人知的经典方法“滑窗”（Sliding window）。滑窗分类器的个数和被测窗的数量呈线性关系，每个尺度的检测在每张图上要１w到10w个窗，多尺度的数量更上升一个数量级。现在的数据集[4]-[6]（PASCAL, ImageNet, MS COCO）还要预测目标的纵横比，将检测空间增加到了每张图100w-1000w个窗。[7]-[11]（[Regionlets for Generic object detection](https://blog.csdn.net/autocyz/article/details/44919725 "Regionlets for Generic object detection in ICCV 2013"), [Rich feature hierarchies for accurate object detection and semantic segmentation](# "R-CNN"), [multibox](#, "Scalable, high quality object detection"), [SPPNet](#, "Spatical pyramid pooling in deep convolutional networks for visual recognition"), [segmentation driven object detection with fisher vectors](#,"segmentation driven object detection with fisher vectors")）易计算和高质量之间难以调和，用一种叫detection proposals的方法可以解决[12]-[15]（[What is an object? CVPR2010](), [Constrained parametric min-cuts for automatic object segmentation CVPR2010](), [Category independent object proposals ECCV2010](), [Segmentation as selective search for object recognition ICCV2011]()）。假设所有关心的目标都会共享一些视觉属性，能和背景区分。因此，可以设计或者训练一个方法输出一系列可能包含这些物体的proposals区域。如果用少量窗口而不是用滑窗也能达到高目标召回率，就能大大提速，使用更复杂的分类器。

detection proposals被用于PASCAL和ImageNet上表现最顶尖的目标检测方法中。除了让更复杂的分类器能使用，它改变了分类器将处理的数据的分布，并可能通过减少假阳性来提高检测质量。

大部分描述生成detection proposals 的文献只用了很局限的评估方法，只在一小部分衡量标准、数据库和方法上评估。本文会在一个统一的框架上比较大部分代码公开的方法。

贡献：第二章 review，分类讨论；第三章 repeatability；第四章研究目标召回率，用 PASCAL VOC2007测试集和ImageNet2013和MS COCO2014验证集，比之前的工作实验范围更广；第五章评估不同 proposal 方法在 DPM, R-CNN, Fast R-CNN上的性能，并发现 AR（平均召回） 和检测性能高度一致。
