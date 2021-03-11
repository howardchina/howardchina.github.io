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
与其它分析不同的是，这里着重提一下onestage的训练逻辑。

看到mmdetection/mmdet/models/detectors/single_stage.py文件的forward_train函数(line68-95)提取特征，特征放到bbox_head.forward_train收集loss。

```python
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses
```

extract_feat包括backbone和neck的forward，只要把输入输出对齐就可以，没有太大难度。而bbox_head比较复杂，后续再展开说。

```python
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
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

这里略提一嘴stage_blocks的初始化来自self.arch_settings。如果depth为50，则初始化的是resnet50。这里讲明白了“stem+n stage+ cls head”的“n stage”部分，而stem部分在resnet中是一个kernel_size=7，stride=2的conv层，这里stem说完了（_make_stem_layer()方法的line559-571）。因为“cls head”部分没用，所以被我们扔了，代码中没有。这里再快速复习一下ResNet的结构：“stem+n stage+ cls head”。

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

 ## neck

mmdetection/mmdet/models/necks/fpn.py文件

总结：neck就是FPN，记住是把c3、c4、c5通过1x1卷积l_conv、上采样和ADD变成了统一通道m3~m5，再通过3x3卷积fpn_conv变成P3~P5，再通过stide=2的3x3卷积加两个特征图最终变成P3~P7。静态地看就是\_\_init\_\_()函数中l_conv和fpn_conv的事。

```python
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
```

下面对代码运行流程进行描述：

1. 将 c3、c4 和 c5 三个特征图全部经过各自 1x1 卷积进行通道变换得到 m3~m5，输出通道统一为 256
2. 从 m5(特征图最小)开始，先进行 2 倍最近邻上采样，然后和 m4 进行 add 操作，得到新的 m4
3. 将新 m4 进行 2 倍最近邻上采样，然后和 m3 进行 add 操作，得到新的 m3
4. 对 m5 和新融合后的 m4、m3，都进行各自的 3x3 卷积，得到 3 个尺度的最终输出 P5～P3
5. 将 c5 进行 3x3 且 stride=2 的卷积操作，得到 P6
6. 将 P6 再一次进行 3x3 且 stride=2 的卷积操作，得到 P7

动态地看是forward()函数（line165-216）里的build top-down path这块涉及到上采样和融合比较重要。

```python
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
```

最后提醒config文件。start_level=1表示把m2扔了，从m3开始；add_extra_convs='on_input'表示从m5开始创建额外的P6、P7。

```python
neck=dict(
    type='FPN',
    # ResNet 模块输出的4个尺度特征图通道数
    in_channels=[256, 512, 1024, 2048],
    # FPN 输出的每个尺度输出特征图通道
    out_channels=256,
    # 从输入多尺度特征图的第几个开始计算
    start_level=1,
    # 额外输出层的特征图来源
    add_extra_convs='on_input',
    # FPN 输出特征图个数
    num_outs=5),
```














[mmdetection-retinanet]: https://zhuanlan.zhihu.com/p/346198300	"轻松掌握 MMDetection 中常用算法(一)：RetinaNet 及配置详解"