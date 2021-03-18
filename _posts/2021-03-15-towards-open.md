---
layout: post
title:  Towards Open World Object Detection
date:   2021-03-18 12:00:00 +0800
categories: [detection]
---

**Paper Reading Note**

URL: https://arxiv.org/abs/2103.02603

## TL;DR
定义“开放世界目标检测”问题，提出评估策略，提供解法：**O**pen Wo**r**ld Object D**e**tector (ORE)。

-----
## Motivation
内在好奇心对学习位置实体有帮助

## 定义问题
   开放世界目标检测
   - 不显式监督学习地识别unknown的物体
   - 随着相关标签出现，逐步学会unknown的物体，而不忘记已知的类型

## Contribution
- 提出Open World Object Detection问题，对现实世界更接近地建模
- 开发ORE方法
- 提出实验对比的方法
- 在增量目标检测上取得SOTA
<!-- - 有效的ORE
- 通过识别和描述unkown实体能降低检测器的逻辑混淆 -->

-----
![20210318213631](/static/img/posts/20210318213631.png "ORE")
## Introduction
通常假设训练时已知所有即将测试的类型。

但是
- 测试图片可以包含unknown类型目标
- 一旦被检测出的unknown类型目标有了标签，模型应该逐步学会新类型

insight：识别unknown的关键是因为好奇
<!-- 
现有方法的问题
- unknown被划为背景
- [7] unknown被划为已知类型
- [42] 评估目标检测预测的不确定性

文章方法在[42]的目标上又前进了一步，一旦知道新类型的类别，就不断学习 -->

## 相关工作

### OpenSet classification
训练集知识不完整，测试遇到新的不认识类型。
- 方法：静态分类器在固定类别的训练集训练
- 缺点：只能识别unknown实体，不能在训练迭代中动态学习

### Open World classification
- 改进：更灵活的设置，让已知类型和unknown类型共存，再次校正类型概率
- 缺点：没有在图像分类benchmark上测试，应用场景局限

### OpenSet Detection
- 改进：正式研究常见目标检测器上的设置，估计目标的不确定性
- 缺点：不能持续调整知识适应动态世界

## Open World Object Detection

- 类型
  - 已知目标
    $$
    \mathcal K^t=\{1,2,...,\rm C\}\subset\mathbb{N}^+
    $$
  - 未知目标
    $$
    \mathcal U=\{\rm C+1,...\}
    $$
- 已知目标类别 $\mathcal K^t$ 的数据库
  $$
  \mathcal D^t=\{{\rm X}^t, {\rm Y}^t\}
  $$
  - M 张输入图像 ${\rm X}^t=\{I_1,...I_M\}$
  - 对应的标签集 ${\rm Y}^t=\{Y_1,...Y_M\}$
  - 每个$Y_i$有K个目标实体$Y_i=\{y_1,y_2,...,y_K\}$
  - $y_k=[l_k,x_k,y_k,w_k,h_k]$ 表示类别 $l_k\in \mathcal K^t$ 和bbox中点宽高
- $\mathcal M_{\rm C}$ 目标检测模型
  - 可识别 $\rm C$ 种已知目标
  - 可将未知目标识别为标签 0
  - 未知实体集合 ${\rm U}^t$
- 更新后 $\mathcal M_{{\rm C}+n}$ 目标检测模型
  - 新增 n 个类别的标签
  - 更新后已知类型
    $$
    \mathcal K^{t+1}=\mathcal K^{t}+\{\rm C+1,...,\rm C +n\}
    $$
