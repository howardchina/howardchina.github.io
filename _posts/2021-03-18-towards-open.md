---
layout: post
title:  Towards Open World Object Detection
date:   2021-03-18 12:00:00 +0800
categories: [detection]
---
**Paper Reading Note**

URL: https://arxiv.org/abs/2103.02603

## TL;DR
定义“开放世界目标检测”问题，提出评估策略，提供基于Faster R-CNN的解法。在原有分类和回归loss之外增加基于 HingeEmbeddingLoss 的聚类loss - clstr loss。

-----
Detectron2 基于Faster R-CNN实现 [Github](https://github.com/JosephKJ/OWOD)


## 定义问题

   开放世界目标检测

   - 不显式监督学习地识别unknown的物体

   - 随着新标签出现，逐步学会unknown的物体，而不削弱已知类型

## Contribution

- 提出Open World Object Detection问题，对现实世界更接近地建模

- 开发ORE方法

- 提出实验对比的方法

- 在增量目标检测上取得SOTA

<!-- - 有效的ORE

- 通过识别和描述unkown实体能降低检测器的逻辑混淆 -->

-----

![20210318213631|690x386,75%](/static/img/posts/20210318213631.png) 
## Introduction

已有方法假设训练时已知所有即将测试的类型。

但是

- 测试图片可以包含unknown类型目标

- 一旦被检测出的unknown类型目标有了新标签，模型应该额外学会新类型，而不从头训练


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

符号定义

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

## 4. ORE: Open World Object Detector

假设隐层空间不同类型间距清晰

- 区分已知实体和未知实体

- 新类型与其他类型不重叠

模仿Faster R-CNN

### 4.1 Contrastive Clustering

$$

\begin{array}{l}

\text { Algorithm } 1 \text { Algorithm COMPUTECLUSTERINGLOSS } \\

\hline \text { Input: Input feature for which loss is computed: } f_{c} ; \text { Feature } \\

\text { store: } \mathcal{F}_{\text {store }} \text { ; Current iteration: } i \text { ; Class prototypes: } \mathcal{P}= \\

\quad\left\{p_{0} \cdots p_{C}\right\} ; \text { Momentum parameter: } \eta \text { . } \\

\text { 1: Initialise } \mathcal{P} \text { if it is the first iteration. } \\

\text { 2: } \mathcal{L}_{\text {cont }} \leftarrow 0 \\

\text { 3: if } i==I_{b} \text { then } \\

\text { 4: } \quad \mathcal{P} \leftarrow \text { class-wise mean of items in } \mathcal{F}_{\text {Store }} \\

\text { 5: } \quad \mathcal{L}_{\text {cont }} \leftarrow \text { Compute using } f_{c}, \mathcal{P} \text { and Eqn. } 1 . \\

\text { 6: else if } i>I_{b} \text { then } \\

\text { 7: } \quad \text { if } i \% I_{p}==0 \text { then } \\

\text { 8: } \quad \mathcal{P}_{\text {new }} \leftarrow \text { class-wise mean of items in } \mathcal{F}_{\text {Store }} . \\

\text { 9: } \quad \mathcal{P} \leftarrow \eta \mathcal{P}+(1-\eta) \mathcal{P}_{\text {new }} \\

\text { 10: } \quad \mathcal{L}_{\text {cont }} \leftarrow \text { Compute using } f_{c}, \mathcal{P} \text { and Eqn. } 1 . \\

\text { 11: return } \mathcal{L}_{\text {cont }}

\end{array}

$$

[code: 聚类loss](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/fast_rcnn.py#L617)


```
class FastRCNNOutputs:
    def losses(self):
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}
```

```    
        losses = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.invalid_class_range,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
        ).losses()
        if input_features is not None:
            # losses["loss_cluster_encoder"] = self.get_ae_loss(input_features)
            losses["loss_clustering"] = self.get_clustering_loss(input_features, proposals)
  ```

算法1 计算聚类loss：原型向量 $p_i$ , 特征向量 $f_c$ ，未知类型label记作 0（实际实现记作N）

- 热身阶段不算loss

- 用特征仓库的均值初始化原型

- 每隔 $I_b$ 轮次更新原型向量

- 聚类loss：目标和同类型原型距离小，和不同类型原型距离大
  基于 HingeEmbeddingLoss [implement](https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html)

  $$
  l_{n}=\left\{\begin{array}{ll}
  x_{n}, & \text { if } y_{n}=1 \\
  \max \left\{0, \Delta-x_{n}\right\}, & \text { if } y_{n}=-1
  \end{array}\right.
  $$

  聚类loss

  $$
  \begin{aligned}
  \mathcal{L}_{\text {cont }}\left(\boldsymbol{f}_{c}\right) &=\sum_{i=0}^{\mathrm{C}} \ell\left(\boldsymbol{f}_{c}, \boldsymbol{p}_{i}\right), \text { where } \\
  \ell\left(\boldsymbol{f}_{c}, \boldsymbol{p}_{i}\right) &=\left\{\begin{array}{ll}
  \mathcal{D}\left(\boldsymbol{f}_{c}, \boldsymbol{p}_{i}\right) & i=c\\
  \max \left\{0, \Delta-\mathcal{D}\left(\boldsymbol{f}_{c}, \boldsymbol{p}_{i}\right)\right\} & \text { otherwise }
  \end{array}\right.
  \end{aligned}
  $$

- 特征仓库 $\mathcal F_{store}$ 大小 CxQ，储存了每个类型Q种分类相关的特征 [update_feature_store](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/fast_rcnn.py#L557)

```
def get_clustering_loss(self, input_features, proposals):
        if not self.enable_clustering:
            return 0

        storage = get_event_storage()
        c_loss = 0
        # if i==I_b
        if storage.iter == self.clustering_start_iter:
            # class_id = -1: get all items in deques
            items = self.feature_store.retrieve(-1)
            for index, item in enumerate(items):
                # no store for unlabeled proposal
                if len(item) == 0:
                    self.means[index] = None
                # for proposals with gt label or pseudo label
                else:
                    mu = torch.tensor(item).mean(dim=0)
                    self.means[index] = mu
            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
            # Freeze the parameters when clustering starts
            # for param in self.ae_model.parameters():
            #     param.requires_grad = False
        elif storage.iter > self.clustering_start_iter:
            if storage.iter % self.clustering_update_mu_iter == 0:
                # Compute new MUs
                items = self.feature_store.retrieve(-1)
                new_means = [None for _ in range(self.num_classes + 1)]
                for index, item in enumerate(items):
                    if len(item) == 0:
                        new_means[index] = None
                    else:
                        new_means[index] = torch.tensor(item).mean(dim=0)
                # Update the MUs
                for i, mean in enumerate(self.means):
                    if(mean) is not None and new_means[i] is not None:
                        self.means[i] = self.clustering_momentum * mean + \
                                        (1 - self.clustering_momentum) * new_means[i]

            c_loss = self.clstr_loss_l2_cdist(input_features, proposals)
        return c_loss
```

### 4.2 Auto-labelling Unknows with RPN

#### 动机：RPN is class agnostic

具体做法：给score高的却不和bbox重合的bbox打标签

- score top-k 的背景proposal当作潜在目标

N+1类，0~N-2是known，N-1是unknow，N是背景

[code: 类型](https://github.com/JosephKJ/OWOD/blob/master/configs/Base-RCNN-C4-OWOD.yaml)

[code: proposal](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/roi_heads.py)

```

if self.enable_thresold_autolabelling:

    matched_labels_ss = matched_labels[sampled_idxs]

    pred_objectness_score_ss = objectness_logits[sampled_idxs]

    # 1) Remove FG objectness score. 2) Sort and select top k. 3) Build and apply mask.

    mask = torch.zeros((pred_objectness_score_ss.shape), dtype=torch.bool)

    pred_objectness_score_ss[matched_labels_ss != 0] = -1

    sorted_indices = list(zip(

        *heapq.nlargest(self.unk_k, enumerate(pred_objectness_score_ss), key=operator.itemgetter(1))))[0]

    for index in sorted_indices:

        mask[index] = True

    gt_classes_ss[mask] = self.num_classes - 1

```

### 4.3 能量函数

把proposal的score和gt class做能量变换（公式4），放入weibull分布拟合。

- weibull函数拟合包 [reliability 链接](https://reliability.readthedocs.io/en/latest/Fitting%20a%20specific%20distribution%20to%20data.html?highlight=Fit_Weibull_3P)

- [code 保存proposals的score到energy](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/roi_heads.py#L481)

$$

E(\boldsymbol{f} ; g)=-T \log \sum_{i=1}^{\mathrm{C}} \exp \left(\frac{g_{i}(\boldsymbol{f})}{T}\right)

$$

- [code 计算lse 公式4](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/engine/train_loop.py#L197)

- 可视化weibull分布结果

![20210319224750|690x475](/static/img/posts/20210319224750.png) 

### 5.1 Open World评价准则

#### 数据划分

基础20类，每个新阶段新增20类 [code setting](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/datasets/coco_utils/balanced_ft.py)

#### 评估方法

$$

\text { Wilderness Impact }(W I)=\frac{P_{\mathcal{K}}}{P_{\mathcal{K} \cup \mathcal{U}}}-1

$$

### 5.2

loss的改动使unknown loss为0 [code](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/fast_rcnn.py#L245)

## Thoughts

- 能量函数，Weibull分布，好像只参与了可视化中，没贡献loss。
- bbox过了ROIPooler之后变成7x7特征，但又mean了一下变成了1x1特征。 [code: mean](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/roi_heads.py#L472)
- 主要创新在store维护的feature队列用于聚类 - 把proposal bbox的feature和gt label存在store，维护聚类。[code: update store](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/fast_rcnn.py#L557)
- loss的最终构成，代码中有变量控制clstr开关：cls，reg，clstr [code](https://github.com/JosephKJ/OWOD/blob/2d33beb02170036e311c76c44c2ac3588bf18841/detectron2/modeling/roi_heads/fast_rcnn.py#L663)