---
layout: post
title:  RetinaNet source code analysis
date:   2021-03-12 03:00:00 +0800
categories: [detection]
---

## 摘要


本文是对 https://zhuanlan.zhihu.com/p/346198300 ,《轻松掌握 MMDetection 中常用算法(一)：RetinaNet 及配置详解》从代码层面上的再解读，便于从代码上改写onestage的目标检测网络。

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

在\_\_init\_\_()函数（line430-447），self.res_layers执行append操作self.stage_blocks的长度次，完成了self.res_layers的初始化
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

## head

RetinaHead: mmdetection/mmdet/models/dense_heads/retina_head.py

AnchorHead: mmdetection/mmdet/models/dense_heads/anchor_head.py

RetinaHead改写了基类AnchorHead的\_\_init\_\_()函数和forward()函数。基类AnchorHead是一层conv构成的cls_conv和conv_reg;

```python
    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)
```

而RetinaHead是在这最后一层之前还加了stacked_convs层的conv。

```python
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
```

RetinaHead的forward_single()函数重载了父类AnchorHead的同名函数，实现前向传播。

```python
    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
```

这个forward_single()函数被父类forward方法调用。需要注意到Returns著名cls_scores和bbox_preds是通过tuple形式组合，而cls_scores和bbox_preds又各自以list形式保存了不同尺度水平anchor的cls和bbox预测值。最终list里的每个元素对应了一个单一尺度水平上的anchor预测值。

```python
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        return multi_apply(self.forward_single, feats)
```

5 个输出 Head 共享所有分类或者回归分支的卷积权重，经过 Head 模块的前向流程输出一共是 5*2 个特征图。

## AnchorGenerator

anchor_generator在mmdetection/mmdet/models/dense_heads/anchor_head.py一共就出现过5次，逐一观察每一次做的事：

1. line37-41， \_\_init\_\_()函数的参数，参数作用在本章第2小节中介绍。

   ```python
                    anchor_generator=dict(
                        type='AnchorGenerator',
                        scales=[8, 16, 32],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[4, 8, 16, 32, 64]),
   ```

   在mmdetection/mmdet/models/dense_heads/retina_head.py文件(line34-39)\_\_init\_\_()函数中被重载和替换：

   ```python
           anchor_generator=dict(
               type='AnchorGenerator',
               octave_base_scale=4,
               scales_per_octave=3,
               ratios=[0.5, 1.0, 2.0],
               strides=[8, 16, 32, 64, 128]),
   ```

   octave_base_scale和scales_per_octave的配合体现在mmdetection/mmdet/core/anchor/anchor_generator.py文件line 96。**octave_scales**取三个值。
   $$
   2^0, 2^{1/3}, 2^{2/3}
   $$
   而scales取3个值[4.0000, 5.0397, 6.3496]，是**octave_scales**的octave_base_scale（4）倍。scales就是。此处self.scales和octave_base_scale+scales_per_octave的组合不能同时使用，原因见mmdetection/mmdet/core/anchor/anchor_generator.py文件line 89，这里不再赘述。在后问**gen_single_level_base_anchors**()的介绍中会举例说明scales、ratios和strides这些参数的作用。

   ```python
               octave_scales = np.array(
                   [2**(i / scales_per_octave) for i in range(scales_per_octave)])
               scales = octave_scales * octave_base_scale
               self.scales = torch.Tensor(scales)
   ```

   如下例所示，scales和注释中的配置等价：

   ```python
   from mmcv.visualization import imshow_bboxes
   import matplotlib.pyplot as plt
   from mmdet.core import build_anchor_generator
   import numpy as np
   
   if __name__ == '__main__':
       anchor_generator_cfg = dict(
           type='AnchorGenerator',
   #         octave_base_scale=4,
   #         scales_per_octave=3,
           scales=[4.0000, 5.0397, 6.3496],
           ratios=[0.5, 1.0, 2.0],
           strides=[8, 16, 32, 64, 128])
       anchor_generator = build_anchor_generator(anchor_generator_cfg)
       # 输出原图尺度上 anchor 坐标 xyxy 左上角格式
       # base_anchors 长度为5，表示5个输出特征图，不同的特征图尺度相差的只是 strides
       # 故我们取 strides=8 的位置 anchor 可视化即可
       base_anchors = anchor_generator.base_anchors[0]
   
       h = 100
       w = 120
       img = np.ones([h, w, 3], np.uint8) * 255
       base_anchors[:, 0::2] += w // 2
       base_anchors[:, 1::2] += h // 2
   
       colors = ['green', 'red', 'blue']
       for i in range(3):
           base_anchor = base_anchors[i::3, :].cpu().numpy()
           imshow_bboxes(img, base_anchor, show=False, colors=colors[i])
       plt.grid()
       plt.imshow(img)
       plt.show()
   ```

   ![20210312170612346]({{ site.url }}/static/img/posts/20210312170612346.png "anchors")

2. line88， **build_anchor_generator**()方法创建了实例。

   ```python
           self.anchor_generator = build_anchor_generator(anchor_generator)
   ```

   mmdetection/mmdet/core/anchor/builder.py文件中，build_anchor_generator通过Registry的'Anchor generator'注册器返回AnchorGenerator类的一个实例。

   ```python
   from mmcv.utils import Registry, build_from_cfg
   
   ANCHOR_GENERATORS = Registry('Anchor generator')
   
   
   def build_anchor_generator(cfg, default_args=None):
       return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)
   
   ```

   深入mmdetection/mmdet/core/anchor/anchor_generator.py文件解读AnchorGenerator。重点关注**gen_base_anchors()**初始化了**self.base_anchors**。

   ```python
   self.base_anchors = self.gen_base_anchors()
   ```

   这个base_anchors在后面的grid_anchors()->**single_level_grid_anchors()**方法中被用到与偏移量相加(line 267)。

   ```python
           all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
   ```

   **gen_base_anchors()**调用**gen_single_level_base_anchors**()方法，对每个base_size大小（base_size定义见mmdetection/mmdet/core/anchor/anchor_generator.py文件line82，执行结果等同于stride，为[4, 8, 16, 32, 64]）的特征图返回一个base_anchors。最终返回多尺度base_anchors的list。

   ```python
       def gen_base_anchors(self):
           """Generate base anchors.
   
           Returns:
               list(torch.Tensor): Base anchors of a feature grid in multiple \
                   feature levels.
           """
           multi_level_base_anchors = []
           for i, base_size in enumerate(self.base_sizes):
               center = None
               if self.centers is not None:
                   center = self.centers[i]
               multi_level_base_anchors.append(
                   self.gen_single_level_base_anchors(
                       base_size,
                       scales=self.scales,
                       ratios=self.ratios,
                       center=center))
           return multi_level_base_anchors
   ```

   **gen_single_level_base_anchors**()方法。ratios是高宽比(h/2)，0.5即h是w的一半。

   ```python
           w = base_size#8
           h = base_size#8
           # 计算高宽比例
           h_ratios = torch.sqrt(ratios)#[0.7071, 1.0000, 1.4142]
           w_ratios = 1 / h_ratios#[1.4142, 1.0000, 0.7071]
           # base_size 乘上宽高比例乘上尺度，就可以得到 n 个 anchor 的原图尺度wh值
           # scales = [4.0000, 5.0397, 6.3496]
           ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)#[45.2548, 57.0175, 71.8376, 32.0000, 40.3175, 50.7968, 22.6274, 28.5088, 35.9188]
           hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)#[22.6274, 28.5088, 35.9188, 32.0000, 40.3175, 50.7968, 45.2548, 57.0175, 71.8376]
           # 得到 x1y1x2y2 格式的 base_anchor 坐标值
           # use float anchor and the anchor's center is aligned with the
           # pixel center
           base_anchors = [
               x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
               y_center + 0.5 * hs
           ]
           # 堆叠起来即可
           base_anchors = torch.stack(base_anchors, dim=-1)
   
           return base_anchors
           """
           tensor([[-22.6274, -11.3137,  22.6274,  11.3137],
           [-28.5088, -14.2544,  28.5088,  14.2544],
           [-35.9188, -17.9594,  35.9188,  17.9594],
           [-16.0000, -16.0000,  16.0000,  16.0000],
           [-20.1587, -20.1587,  20.1587,  20.1587],
           [-25.3984, -25.3984,  25.3984,  25.3984],
           [-11.3137, -22.6274,  11.3137,  22.6274],
           [-14.2544, -28.5088,  14.2544,  28.5088],
           [-17.9594, -35.9188,  17.9594,  35.9188]])
           """
   ```

   简单来说就是：假设一共 m 个输出特征图

   - 遍历 m 个输出特征图，在每个特征图的 (0,0) 或者说原图的 (0,0) 坐标位置生成 `base_anchors`，注意 `base_anchors` 不是特征图尺度，而是原图尺度
   - 遍历 m 个输出特征图中每个特征图上每个坐标点，将其映射到原图坐标上
   - 原图坐标点加上 `base_anchors`，就可以得到特征图每个位置的对应到原图尺度的 anchor 列表，anchor 列表长度为 m

3. line 91， self.num_anchors统计base anchors个数，比如这里有9个。

   ```python
           self.num_anchors = self.anchor_generator.num_base_anchors[0]
   ```

   每个点产生这9个anchor，对应到\_init\_layers()就是conv的通道为self.num_anchors * self.cls_out_channels（coco数据集80类，每个点就是9*80个类型）以及self.num_anchors * 4（每个点的9个anchors一共就有9*4个值要确定）(line 94-98)。

   ```python
       def _init_layers(self):
           """Initialize layers of the head."""
           self.conv_cls = nn.Conv2d(self.in_channels,
                                     self.num_anchors * self.cls_out_channels, 1)
           self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 4, 1)
   ```

   

4. line 158，self.anchor_generator.grid_anchors得到5个特征图上的anchors。

   ```python
           multi_level_anchors = self.anchor_generator.grid_anchors(
               featmap_sizes, device)
   ```

   需要了解

5. 

## assigner

ref: mmdetection/mmdet/models/dense_heads/anchor_head.py (line79)

define: mmdetection/mmdet/core/bbox/assigners/max_iou_assigner.py

把ref放在前面是为了凸显assigner在调用中的顺序；先在AnchorHead中初始化(line79)，

```python
self.assigner = build_assigner(self.train_cfg.assigner)
```

然后再_get_targets_single()方法(line208)中被调用。

```python
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
```

逻辑层面上，assigner要根据cfg配置返回一个合适的AssignResult类的实例。

```python
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
```

总结：AnchorHead类初始化assigner实例->AnchorHead类_get_targets_single()方法->MaxIoUAssigner类重载父类BaseAssigner类assign()方法

## loss

代码上需要了解single_stage.py内定义函数的调用层级关系。

以训练过程为例：

1.single_stage.py文件forward_train方法(line 93)调用bbox_head.forward_train方法返回losses给上一层反向传播。

```python
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
```

2.bbox_head.forward_train继承自父类，在mmdetection/mmdet/models/dense_heads/base_dense_head.py文件中(line22)定义。

```python
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
```

2.1.先执行bbox_head的**forward**函数(2.1.1)。

```python
outs = self(x)
```

2.1.1.**forward**函数在mmdetection/mmdet/models/dense_heads/anchor_head.py文件(line122)定义。此处衔接head章节，已经讨论清楚了。总结就是**forward**->**multi_apply**->**forward_single**的流程，目前没有涉及assigner，所以assigner是在**loss**在调用。

2.2.调用**loss**()函数(2.2.1)，把上一步forward的结果和gt比对。

```python
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
```

2.2.1.RetinaHead调用的是继承自父类AnchorHead的loss函数，在mmdetection/mmdet/models/dense_heads/anchor_head.py文件(line420)。

```python
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        # 每个scale level中featmap的高度和宽度
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        # 获取特征图大小的anchor
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
```

a.知道每个scale level中featmap的高度和宽度

```python
        # 每个scale level中featmap的高度和宽度
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
```

b.获取featmap中的anchor位置

```python
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
```

​    b.1. get_anchors()函数定义在mmdetection/mmdet/models/dense_heads/anchor_head.py文件line141，在章节anchor generator中已经讲清楚。

```python
    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
```


2.3.返回losses（和调用get_bboxes得到初步预测结果，onestage的方法如RetinaNet没有这一步）。

```python
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
```

