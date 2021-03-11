---
layout: post
title:  RetinaNet source code analysis
date:   2021-03-12 03:00:00 +0800
categories: [detection]
---

## 摘要


本文是对[mmderection-retinanet][mmderection-retinanet] 从代码层面上的再解读，便于从代码上改写onestage的目标检测网络。

其简要网络结构图如下：

![20210312002436]({{ site.url }}/static/img/posts/20210312002436.png "RetinaNet")

## 主体结构 

RetinaNet包括：

* backbone: ResNet, mmdetection/mmdet/models/backbones/resnet.py
* neck: FPN, 
* bbox_head: RetinaHead
  * anchor_generator
  * bbox_coder
  * loss_cls
  * loss_bbox
* assigner
* postprocessing

我们看一下配置代码中各个模块的位置关系：   

```python
# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    #postprocessing
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100)

```
## backbone

看下mmdet/models/backbones/resnet.py文件 forward函数中提取特征的关键部分。

return的outs即为多尺度的特征，append(x)每次套了一个尺度。所以疑问是循环中self.res_layers套了几层？self.res_layers在哪儿产生的？

```python
outs = []
for i, layer_name in enumerate(self.res_layers):
res_layer = getattr(self, layer_name)
x = res_layer(x)
if i in self.out_indices:
outs.append(x)
return tuple(outs)
```

在\_\_init\_\_()函数（大约430-447行），self.res_layers执行append操作self.stage_blocks的长度次，完成了self.res_layers的初始化
```python
            self.res_layers = []
            for i, num_blocks in enumerate(self.stage_blocks):
                stride = strides[i]
                dilation = dilations[i]
                dcn = self.dcn if self.stage_with_dcn[i] else None
                if plugins is not None:
                    stage_plugins = self.make_stage_plugins(plugins, i)
                else:
                    stage_plugins = None
                planes = base_channels * 2**i
                res_layer = self.make_res_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    plugins=stage_plugins)
                self.inplanes = planes * self.block.expansion
                layer_name = f'layer{i + 1}'
                self.add_module(layer_name, res_layer)
                self.res_layers.append(layer_name)
```

这里略提一嘴stage_blocks的初始化来自self.arch_settings。如果depth为50，则初始化的是resnet50。这里讲明白了“stem+n stage+ cls head”的“n stage”部分，而stem部分在resnet中是一个kernel_size=7，stride=2的conv层，这里stem说完了（_make_stem_layer()方法的line559-571）。因为“cls head”部分没用，所以被我们扔了，代码中没有。这里再快速复习一下ResNet的结构。

![20210312001916]({{ site.url }}/static/img/posts/20210312001916.png "resnet")

(ResNet paper)[https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf]
      
```python
      arch_settings = {
      	18: (BasicBlock, (2, 2, 2, 2)),
      	34: (BasicBlock, (3, 4, 6, 3)),
      	50: (Bottleneck, (3, 4, 6, 3)),
      	101: (Bottleneck, (3, 4, 23, 3)),
      	152: (Bottleneck, (3, 8, 36, 3))
      }
      ...
      num_stages=4
      ...
      self.block, stage_blocks = self.arch_settings[depth]
      self.stage_blocks = stage_blocks[:num_stages]
```

 


















[mmdetection-retinanet]: https://zhuanlan.zhihu.com/p/346198300	"轻松掌握 MMDetection 中常用算法(一)：RetinaNet 及配置详解"