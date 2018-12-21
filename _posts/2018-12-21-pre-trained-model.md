---
layout: post
title: Ranked Pre-trained models on Github
date:   2018-12-21 22:23:00 +0800
categories: [model]
---

## 较新和较早的网络

参照：[Github:pretrained-models](https://github.com/Cadene/pretrained-models.pytorch) 

| Model           | Version                               | Acc@1 | Acc@5 |
| --------------- | ------------------------------------- | ----- | ----- |
| PNASNet-5-Large | [Tensorflow]## Evaluation on imagenet |       |       |

### Accuracy on validation set (single model)

Results were obtained using (center cropped) images of the same size than during the training process.

| Model                                                        | Version                                                      | Acc@1  | Acc@5  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------ | ------ |
| PNASNet-5-Large                                              | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 82.858 | 96.182 |
| [PNASNet-5-Large](https://github.com/Cadene/pretrained-models.pytorch#pnasnet) | Our porting                                                  | 82.736 | 95.992 |
| NASNet-A-Large                                               | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 82.693 | 96.163 |
| [NASNet-A-Large](https://github.com/Cadene/pretrained-models.pytorch#nasnet) | Our porting                                                  | 82.566 | 96.086 |
| SENet154                                                     | [Caffe](https://github.com/hujie-frank/SENet)                | 81.32  | 95.53  |
| [SENet154](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 81.304 | 95.498 |
| PolyNet                                                      | [Caffe](https://github.com/CUHK-MMLAB/polynet)               | 81.29  | 95.75  |
| [PolyNet](https://github.com/Cadene/pretrained-models.pytorch#polynet) | Our porting                                                  | 81.002 | 95.624 |
| **InceptionResNetV2**                                        | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.4   | 95.3   |
| **InceptionV4**                                              | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.2   | 95.3   |
| [SE-ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 80.236 | 95.028 |
| SE-ResNeXt101_32x4d                                          | [Caffe](https://github.com/hujie-frank/SENet)                | 80.19  | 95.04  |
| [InceptionResNetV2](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting                                                  | 80.170 | 95.234 |
| [InceptionV4](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting                                                  | 80.062 | 94.926 |
| [DualPathNet107_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.746 | 94.684 |
| ResNeXt101_64x4d                                             | [Torch7](https://github.com/facebookresearch/ResNeXt)        | 79.6   | 94.7   |
| [DualPathNet131](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.432 | 94.574 |
| [DualPathNet92_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.400 | 94.620 |
| [DualPathNet98](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 79.224 | 94.488 |
| [SE-ResNeXt50_32x4d](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 79.076 | 94.434 |
| SE-ResNeXt50_32x4d                                           | [Caffe](https://github.com/hujie-frank/SENet)                | 79.03  | 94.46  |
| [Xception](https://github.com/Cadene/pretrained-models.pytorch#xception) | [Keras](https://github.com/keras-team/keras/blob/master/keras/applications/xception.py) | 79.000 | 94.500 |
| [ResNeXt101_64x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting                                                  | 78.956 | 94.252 |
| [Xception](https://github.com/Cadene/pretrained-models.pytorch#xception) | Our porting                                                  | 78.888 | 94.292 |
| ResNeXt101_32x4d                                             | [Torch7](https://github.com/facebookresearch/ResNeXt)        | 78.8   | 94.4   |
| SE-ResNet152                                                 | [Caffe](https://github.com/hujie-frank/SENet)                | 78.66  | 94.46  |
| [SE-ResNet152](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 78.658 | 94.374 |
| ResNet152                                                    | [Pytorch](https://github.com/pytorch/vision#models)          | 78.428 | 94.110 |
| [SE-ResNet101](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 78.396 | 94.258 |
| SE-ResNet101                                                 | [Caffe](https://github.com/hujie-frank/SENet)                | 78.25  | 94.28  |
| [ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting                                                  | 78.188 | 93.886 |
| FBResNet152                                                  | [Torch7](https://github.com/facebook/fb.resnet.torch)        | 77.84  | 93.84  |
| SE-ResNet50                                                  | [Caffe](https://github.com/hujie-frank/SENet)                | 77.63  | 93.64  |
| [SE-ResNet50](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting                                                  | 77.636 | 93.752 |
| [DenseNet161](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.560 | 93.798 |
| [ResNet101](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.438 | 93.672 |
| [FBResNet152](https://github.com/Cadene/pretrained-models.pytorch#facebook-resnet) | Our porting                                                  | 77.386 | 93.594 |
| [InceptionV3](https://github.com/Cadene/pretrained-models.pytorch#inception) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.294 | 93.454 |
| [DenseNet201](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 77.152 | 93.548 |
| [DualPathNet68b_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 77.034 | 93.590 |
| [CaffeResnet101](https://github.com/Cadene/pretrained-models.pytorch#caffe-resnet) | [Caffe](https://github.com/KaimingHe/deep-residual-networks) | 76.400 | 92.900 |
| [CaffeResnet101](https://github.com/Cadene/pretrained-models.pytorch#caffe-resnet) | Our porting                                                  | 76.200 | 92.766 |
| [DenseNet169](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 76.026 | 92.992 |
| [ResNet50](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 76.002 | 92.980 |
| [DualPathNet68](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting                                                  | 75.868 | 92.774 |
| [DenseNet121](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 74.646 | 92.136 |
| [VGG19_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 74.266 | 92.066 |
| NASNet-A-Mobile                                              | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 74.0   | 91.6   |
| [NASNet-A-Mobile](https://github.com/veronikayurchuk/pretrained-models.pytorch/blob/master/pretrainedmodels/models/nasnet_mobile.py) | Our porting                                                  | 74.080 | 91.740 |
| [ResNet34](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 73.554 | 91.456 |
| [BNInception](https://github.com/Cadene/pretrained-models.pytorch#bninception) | Our porting                                                  | 73.524 | 91.562 |
| [**VGG16_BN**](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 73.518 | 91.608 |
| [VGG19](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 72.080 | 90.822 |
| [**VGG16**](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 71.636 | 90.354 |
| [VGG13_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 71.508 | 90.494 |
| [VGG11_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 70.452 | 89.818 |
| [ResNet18](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 70.142 | 89.274 |
| [VGG13](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 69.662 | 89.264 |
| [VGG11](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 68.970 | 88.746 |
| [SqueezeNet1_1](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 58.250 | 80.800 |
| [SqueezeNet1_0](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 58.108 | 80.428 |
| [Alexnet](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models)          | 56.432 | 79.194 |

---

## Pre-trained Models

参照：[model and checkpoints on tf-slim](https://github.com/tensorflow/models/tree/master/research/slim)

Neural nets work best when they have many parameters, making them powerful function approximators. However, this means they must be trained on very large datasets. Because training models from scratch can be a very computationally intensive process requiring days or even weeks, we provide various pre-trained models, as listed below. These CNNs have been trained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset.

In the table below, we list each model, the corresponding TensorFlow model file, the link to the model checkpoint, and the top 1 and top 5 accuracy (on the imagenet test set). Note that the VGG and ResNet V1 parameters have been converted from their original caffe formats ([here](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014) and [here](https://github.com/KaimingHe/deep-residual-networks)), whereas the Inception and ResNet V2 parameters have been trained internally at Google. Also be aware that these accuracies were computed by evaluating using a single image crop. Some academic papers report higher accuracy by using multiple crops at multiple scales.

| Model                                                        | TF-Slim File                                                 | Checkpoint                                                   | Top-1 Accuracy | Top-5 Accuracy |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- | -------------- |
| [Inception V1](http://arxiv.org/abs/1409.4842v1)             | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py) | [inception_v1_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz) | 69.8           | 89.6           |
| [Inception V2](http://arxiv.org/abs/1502.03167)              | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py) | [inception_v2_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz) | 73.9           | 91.8           |
| [Inception V3](http://arxiv.org/abs/1512.00567)              | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) | [inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz) | 78.0           | 93.9           |
| [Inception V4](http://arxiv.org/abs/1602.07261)              | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) | [inception_v4_2016_09_09.tar.gz](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) | 80.2           | 95.2           |
| [Inception-ResNet-v2](http://arxiv.org/abs/1602.07261)       | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py) | [inception_resnet_v2_2016_08_30.tar.gz](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz) | 80.4           | 95.3           |
| [ResNet V1 50](https://arxiv.org/abs/1512.03385)             | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) | [resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) | 75.2           | 92.2           |
| [ResNet V1 101](https://arxiv.org/abs/1512.03385)            | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) | [resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) | 76.4           | 92.9           |
| [ResNet V1 152](https://arxiv.org/abs/1512.03385)            | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) | [resnet_v1_152_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz) | 76.8           | 93.2           |
| [ResNet V2 50](https://arxiv.org/abs/1603.05027)^            | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) | [resnet_v2_50_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz) | 75.6           | 92.8           |
| [ResNet V2 101](https://arxiv.org/abs/1603.05027)^           | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) | [resnet_v2_101_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz) | 77.0           | 93.7           |
| [ResNet V2 152](https://arxiv.org/abs/1603.05027)^           | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) | [resnet_v2_152_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz) | 77.8           | 94.1           |
| [ResNet V2 200](https://arxiv.org/abs/1603.05027)            | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py) | [TBA](https://github.com/tensorflow/models/blob/master/research/slim) | 79.9*          | 95.2*          |
| [VGG 16](http://arxiv.org/abs/1409.1556.pdf)                 | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) | [vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) | 71.5           | 89.8           |
| [VGG 19](http://arxiv.org/abs/1409.1556.pdf)                 | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py) | [vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) | 71.1           | 89.8           |
| [MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf) | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) | [mobilenet_v1_1.0_224.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | 70.9           | 89.9           |
| [MobileNet_v1_0.50_160](https://arxiv.org/pdf/1704.04861.pdf) | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) | [mobilenet_v1_0.50_160.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz) | 59.1           | 81.9           |
| [MobileNet_v1_0.25_128](https://arxiv.org/pdf/1704.04861.pdf) | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) | [mobilenet_v1_0.25_128.tgz](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz) | 41.5           | 66.3           |
| [MobileNet_v2_1.4_224^*](https://arxiv.org/abs/1801.04381)   | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) | [mobilenet_v2_1.4_224.tgz](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz) | 74.9           | 92.5           |
| [MobileNet_v2_1.0_224^*](https://arxiv.org/abs/1801.04381)   | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py) | [mobilenet_v2_1.0_224.tgz](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) | 71.9           | 91.0           |
| [NASNet-A_Mobile_224](https://arxiv.org/abs/1707.07012)#     | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py) | [nasnet-a_mobile_04_10_2017.tar.gz](https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz) | 74.0           | 91.6           |
| [NASNet-A_Large_331](https://arxiv.org/abs/1707.07012)#      | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py) | [nasnet-a_large_04_10_2017.tar.gz](https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz) | 82.7           | 96.2           |
| [PNASNet-5_Large_331](https://arxiv.org/abs/1712.00559)      | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py) | [pnasnet-5_large_2017_12_13.tar.gz](https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_large_2017_12_13.tar.gz) | 82.9           | 96.2           |
| [PNASNet-5_Mobile_224](https://arxiv.org/abs/1712.00559)     | [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/pnasnet.py) | [pnasnet-5_mobile_2017_12_13.tar.gz](https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_mobile_2017_12_13.tar.gz) | 74.2           | 91.9           |

^ ResNet V2 models use Inception pre-processing and input image size of 299 (use `--preprocessing_name inception --eval_image_size 299` when using `eval_image_classifier.py`). Performance numbers for ResNet V2 models are reported on the ImageNet validation set.

(#) More information and details about the NASNet architectures are available at this [README](https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/README.md)

All 16 float MobileNet V1 models reported in the [MobileNet Paper](https://arxiv.org/abs/1704.04861) and all 16 quantized [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) compatible MobileNet V1 models can be found [here](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet_v1.md).

(^#) More details on MobileNetV2 models can be found [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

(*): Results quoted from the [paper](https://arxiv.org/abs/1603.05027).