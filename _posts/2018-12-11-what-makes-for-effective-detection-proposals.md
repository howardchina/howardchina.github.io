---
layout: post
title: PAMI2015 What makes for effective detection proposals?
date:   2018-12-11 16:12:00 +0800
categories: [proposals]
---

### What makes for effective detection proposals?

### 什么造就了高效的detection proposals?

作者：Jan Hosang<sup>1</sup>, Rodrigo Benenson<sup>1</sup>, Piotr Dollar<sup>2</sup>和Bernt Schiele<sup>1</sup>

单位：１马克斯普朗克信息学研究所，Max Planck Institute for Informatics；２Facebook 人工智能研究，Facebook AI Research (FAIR)

**摘要**：眼下最好的目标检测器都用了detection proposals来引导目标搜索，因此避免了在图上滑窗搜索这种开销大的方式。尽管detection proposals流行且用途广泛，但仍不确定用它们检测目标时用了什么权衡策略。本文深入分析12个proposals方法以及4种baselines，考虑在PASCAL, ImageNet和MS COCO数据库上的proposal repeatability（产生proposal的稳定性）和金标准召回率，以及对DPM（2010,　discriminatively trained part-based models）, R-CNN和Fast R-CNN检测性能的影响。本文的分析说明了对目标检测而言提高proposal localisation accuracy和提高召回率一样重要。本文提出一种新颖的测量方法，平均召回率（AR），奖励高召回率和好的localisation，与detection的性能出奇地一致。本文的结论说明了现有方法的通常存在的强项和缺点，为选择和调试方法提供了见解和测量方法。

**关键词**：Computer Vision, objection detection, detection proposals.

[Github](https://github.com/hosang/detection-proposals)
[Offical Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/how-good-are-detection-proposals-really/)

![Figure 1 AR on the PASCAL VOC 2007 test set]({{ site.url }}/static/img/posts/csm_AR_pascal_17ce0d1c62.png "Figure 1 AR on the PASCAL VOC 2007 test set")

![Figure 2 AR on the ImageNet 2013 validation set]({{ site.url }}/static/img/posts/csm_AR_imagenet_cf092e0852.png "Figure 2 AR on the ImageNet 2013 validation set")

![Figure 3 AR on the COCO 2014 validation set]({{ site.url }}/static/img/posts/csm_AR_coco_abb533c563.png "Figure 3 AR on the COCO 2014 validation set")

---

**1.简介**

广为人知的经典方法“滑窗”（Sliding window）。滑窗分类器的个数和被测窗的数量呈线性关系，每个尺度的检测在每张图上要１w到10w个窗，多尺度的数量更上升一个数量级。现在的数据集[4]-[6]（PASCAL, ImageNet, MS COCO）还要预测目标的纵横比，将检测空间增加到了每张图100w-1000w个窗。[7]-[11]（[Regionlets for Generic object detection](https://blog.csdn.net/autocyz/article/details/44919725 "Regionlets for Generic object detection in ICCV 2013"), [Rich feature hierarchies for accurate object detection and semantic segmentation](# "R-CNN"), [multibox](#, "Scalable, high quality object detection"), [SPPNet](#, "Spatical pyramid pooling in deep convolutional networks for visual recognition"), [segmentation driven object detection with fisher vectors](#,"segmentation driven object detection with fisher vectors")）易计算和高质量之间难以调和，用一种叫detection proposals的方法可以解决[12]-[15]（[What is an object? CVPR2010](), [Constrained parametric min-cuts for automatic object segmentation CVPR2010](), [Category independent object proposals ECCV2010](), [Segmentation as selective search for object recognition ICCV2011]()）。假设所有关心的目标都会共享一些视觉属性，能和背景区分。因此，可以设计或者训练一个方法输出一系列可能包含这些物体的proposals区域。如果用少量窗口而不是用滑窗也能达到高目标召回率，就能大大提速，使用更复杂的分类器。

detection proposals被用于PASCAL和ImageNet上表现最顶尖的目标检测方法中。除了让更复杂的分类器能使用，它改变了分类器将处理的数据的分布，并可能通过减少假阳性来提高检测质量。

大部分描述生成detection proposals 的文献只用了很局限的评估方法，只在一小部分衡量标准、数据库和方法上评估。本文会在一个统一的框架上比较大部分代码公开的方法。

贡献：第二章 review，分类讨论；第三章 repeatability；第四章研究目标召回率，用 PASCAL VOC2007测试集和ImageNet2013和MS COCO2014验证集，比之前的工作实验范围更广；第五章评估不同 proposal 方法在 DPM, R-CNN, Fast R-CNN上的性能，并发现 AR（平均召回） 和检测性能高度一致。

**2.检测 proposal 方法**

类似于"兴趣点"检测器[30][31]。"兴趣点"找出图上最显著的位置，简化了后续任务的计算量。随着计算速度的提升，**Dense Interest points[32]** 投入使用;而另一种趋势，同样 **dense** 的 sliding window 被 proposal 取代了。所以本文旨在理解 proposal 在保持精度的情况下取代了 sliding window 。

目标备选（object proposals）的两类：**grouping** 和 **window scoring** 将在第2章阐述。

*表1* 记录了对12种挑选出的方法（外加4个baseline）做的充分实验。

![1544846034288]({{site.url}}/static/img/posts/1544846034288.png)

本文关注的 proposal 方法都是 **类别无关的，只针对单张图片且只检测bounding box** 。对那些输出的分割结果的，也转换成了bounding box来对比。不考虑用于视频或者基于模板信息的方法。

**2.1 Grouping 的方法**

旨在产生多个和目标相关的（有可能重叠的）分割结果。
最简单的方法就是用 **层次图像分割** 算法的输出，例如[34]用了[35]的输出。为了使得结果更加多样化，例如[19][26][29]用了多个低阶分割，或者[26]从 **过分割** 开始逐步 **随机合并** 。合并依据很多特征，比如 **超像素的形状和外观、边界估算** （[35][36]通常是这样做的）。

根据产生 proposal 的方式，分为三种：超像素（superpixels, SP）[37]、多种子的多图割（multiple graph cut, GC）、边缘轮廓（edge contours, EC）[35][36]。SP的结果都是segmentation，也都转化为 bounding box 来对比。

#### superpixels
* **SelectiveSearch** <sup>SP</sup> [15][29] 贪心合并超像素。不会学习到参数，只利用人工设计的特征和相似度方程，合并超像素。用于R-CNN和Faster R-CNN [8][16].
* **RandomizedPrim's** <sup>SP</sup> [26] 用了类似SelectiveSearch的特征，提出一种随机超像素合并策略，其中用到了学习到的参数。速度提升。
* **Rantalankila** <sup>SP</sup> [27] 和 SelectiveSearch 类似的超像素合并策略，用了不同的特征。后续作为CPMC（见下方）的种子点来产生更多 proposals。
* Chang <sup>SP</sup> [38] 结合显著性和图模型上的目标检测，合并超像素，产生图或者背景。

#### graph-cut
* **CPMC** <sup>GC</sup> [13][19] 不用初分割，在像素上用一些种子和unaries做GC。用一个很大的特征池对分割结果排序。
* **Endres** <sup>GC</sup> [14][21] 从遮挡边界上构建层次分割，不同的种子，用图割解决，生成分割。 基于大量依据对 proposals 排序，激励多样化的结果。
* **Rigor** <sup>GC</sup> [28] CPMC 改良的变体，通过重用多个图割问题的计算、采用[36][39]的快速边缘检测器，极大加速了计算。

#### edge contours
* **Geodesic** <sup>EC</sup> [22] 以过分割的[36]的结果开始。用分类器布置种子来做测地距离变换。每个距离变换的水平集定义了 proposals 的分割。
* **MCG** <sup>EC</sup> [23] 在[36]之上，提出一种快速计算多尺度层次分割的算法。根据边的强度合并分割，用大小、位置、形状、边强估算目标的可能性，对结果排序。

**2.2 Window scoring 的方法**

对每个 window 包含目标的可能性打分。通常只产生边界框（bounding box），更快。定位准确度更低，但有的方法改进了定位。

* **Objectness** [12][24] proposal 鼻祖。从图片的显著位置提出若干初始 proposals ，再进一步依据一些线索对其打分，比如颜色、边缘、位置、大小、超像素跨界（superpixel straddling）。
* **Rahtu** [25] 先从单一、两个、三个超像素，以及多个随机采样的框中构造一个很大的 proposal 区域。再次用到了 *Objectness* 的打分策略，也做了些许改进。[40] 添加了低阶特征，强调了非极大值抑制（non-maximum suppression, NMS）。
* **Bing** [18] 在滑窗上加了在边缘特征上预训练过的线性分类器。非常快速的（1ms/image on CPU ）类别无关的检测器。但是，[41]（ CrackingBing ）发现不用分类器，得到近似性能。
* **EdgeBoxes** <sup>EC</sup> [20] 从粗糙的滑窗模式开始，建立目标边界估计（采用[36][42]结构化的决策森林）,后续再提高定位精度。不用通过学习得到参数。调整滑窗模式的密度和非极大值抑制，来得到不同重叠的阈值（在第5章）。
* Feng [43] 搜索显著图像内容。提出一种显著性测度，由图像的剩余部分产生显著目标。滑窗，依据显著性线索打分。
* Zhang [44] 利用简单梯度特征训练一个级联的SVM来ranking。第一步，对每个尺度和纵横比做一个单独的分类器;第二步，对前面得到的 proposals ranking。所有的 SVM 都是用结构化输出学习训练的，以对于那些与目标重叠得更多的 window 打更高分。但是这种级联用了同样类别的训练集和测试集，不能确定跨类别的泛化性能。
* RandomizedSeeds [45] 用多个随机 SEED 超像素映射[46] 对每个候选窗打分。简单打分方式和 superpixel straddling 类似，没用额外线索。多超像素映射（ multiple superpixel maps ）大大提高召回率。

**2.3 其他**

* ShapeSharing [47] 无参数、数据驱动。用边缘匹配，将目标形状从模板迁移到测试图像。然后利用图割，将得到的区域合并和调整。
* Multibox [9][48] 无需滑窗，用 **神经网络** 回归得到固定数目的 proposals 。每个 proposals 都有独特的位置偏好，增加了 proposals 位置的多样性。在 ImageNet 上表现 top 。

**2.4 Baseline**

类别无关
* **Uniform**
均匀采样，边界框中心位置、平方根面积、 log 宽高比，在PASCAL VOC 2007上测试，去除0.5%的极值（数据覆盖率99%）。
* **Gaussian**
先从训练集上计算均值、协方差，再从分布中采样。多元高斯分布，边界框中心位置、平方根面积、 log 宽高比。
* **SlidingWindow**
像滑窗目标检测器一样，把窗口放置网格上。窗口数量由窗的宽高决定，均匀放置每种大小的窗口。这个过程同 Bing 的实现[18][41]。
* **Superpixels**
本文5种方法用到了[37]，所以直接用其做baseline：每个低阶分割都当做一个检测proposal。超像素方法召回率的下限。

除了 Superpixels 以外的 Baselines 都和图像内容无关。确定图像大小，SlidingWindow 就是确定的，Uniform 和 Gaussian 是随机的。

**2.5 Proposals 对比 cascades**

很多 proposals 方法（显然 SelectivweSearch 和 EdgeBoxes 除外）通过 **图像特征** 生成候选窗口，这可以认为是个具有 **判别性** 的过程——给定一些特征后，算法能快速判断窗口是否值得检测。因此，proposal 方法就和 cascades 有关联[2][49]-[51]，也就是用一个 **快速而不准确的分类器** 丢弃绝大部分不可靠的 proposals 。但有时候对于特定类别的检测， cascades 也可以设置一些类别[52][53]。

通常的 cascade 和 proposal 的关键区别在于，后者要泛化超出训练中观察到的目标类别。什么是使 proposal 在高度区别的训练后仍然泛化到没见过的类别的关键？有一个的假设，训练一个类别足够多的分类器对于 proposal 的泛化能力非常重要。此外，分类器的区分能力通常是 **局限的** (eg: Bing, Zhang)，有助于避免过拟合，并学到一些目标粗糙的属性（比如“物体是圆的”），在[54]中也提到过。通过一些有额外类别的数据库来测试 proposal 方法的泛化能力（第4章）。

**2.6 控制 proposals 的数量**

表1 ，12个方法+4 baselines 的广泛的对比比较。控制 proposals 的数量便于比较。默认情况下，待测方法的 proposals 数量从10<sup>2</sup> 到10<sup>5</sup> 不等。此外，有的输出ranking或者scored 的 proposals ，其它没有。更多的 proposals 意味着更大的几率获得更高的 Recall 。如下，控制生成的 proposals 数量。

Objectness, CPMC, Endres, SelectiveSearch, Rahtu, Bing, MCG, EdgeBoxes 提供了scored 或 sorted proposals，取前k个。
Rantalankila, Rigor, Geodesic 没有提供直接控制k，但可以间接通过了一个参数控制它。在一个子集的图像上实验，线性分析了参数设置和k。
RandomizedPrim's没有任何 proposals 控制方法，随机采样其中k个 proposals。

后处理，上述方法产生了重复的 proposals ，都被去重。

**3.Proposal repeatability**
Proposal 的可重复能力

proposal 会改变正负窗口的分布，在第4章讨论在正样本中的 proposal 与 groundtruth 的重叠率。本章分析负样本的分布：如果 proposal 方法不能在 **相似的图像内容** 上，比如没有目标的内容或者有部分目标的内容上，**产生一致的窗口**，分类器就没办法对负样本打分。考虑极端情况，训练集中的 proposal 必包含目标，而测试集的 proposal 有目标和负样本，这样训练出来的分类器对于负样本的打分没有意义。因此认为，在背景上产生一致分布的 proposal 也会影响检测。

repeatability 就是在相似图像内容上产生 proposals 的性质。直觉上，不同图片上应该产生相似内容的 proposals 。设计实验，比较 proposals 和在修改过的图片上产生的 proposals 。因为PASCAL VOC 没有合适的图像，另一个数据库[31]只有54张图和非常少的目标，所以选择将 **合成转换** 用于 PASCAL 图像。

**3.1 repeatability 的衡量规则**

依据来自[31]，评估兴趣点的 repeatability。PASCAL VOC 2007的测试集加干扰：模糊、旋转、缩放、亮度、JPEG 压缩和椒盐噪声（见图3-图4）。

每一对参考图和干扰图中用给定方法产生1000个窗口。将干扰图的 proposals 投影回参考图，再与参考图的 proposals 匹配（实现时，如果 proposals 旋转后中心超出了图像范围，剔除这样的 proposals ，不用于匹配）。用 IoU (intersection over union 并中交) 作为标准，用贪心法两两匹配（提高匹配效率）。对给定的匹配结果，做图表示取每种 IoU 阈值时的 recall，并定义 repeatability 为 “recall 对比 IoU 阈值”曲线（IoU 在0和1之间）的面积，类似于参考图的 proposals 的平均最好重叠（ABO，见附录A）。那些能得到相似位置、高 IoU，也就是相似图像内容的窗口的方法更有可重复性，因此曲线之下的面积更大。

这样的 proposal 匹配有个问题，大的窗口比小的窗口更易被匹配（同样的干扰对小窗影响更大）。需要考虑这个重要的影响，因为不同的方法有不同的 proposal 窗口大小分布，如图5a 所示。为减轻这个现象的影响，将窗口根据其面积的大小分为10组，分别计算每个窗大小下的recall versus IoU 曲线的面积。图5b 展示了这10组在轻微模糊干扰的效果。大的 proposals 有更高的 repeatability 。在所有图5 的实验中，展示都是10组大小平均后的值。

省略最慢的 CPMC 和 Endres （每个干扰在整个 PASCAL 上要跑约50次）。

![1544878922801]({{site.url}}/static/img/posts/1544878922801.png)
