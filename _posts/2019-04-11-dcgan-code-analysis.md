---
layout: post
title:  DCGAN code analysis
date:   2019-04-11 15:08:00 +0800
categories: [GAN]
---

Git: https://github.com/soumith/dcgan.torch.git

running environment: torch, cuda, cudnn (install these packages as the tutorial suggested)

---

# Training

file: main.lua

image size: (3, 64, 64)

**nz**: *100* dim for z

**main** -> **data** -> **donkey_folder**

## netG

local netG = nn.Sequential()

-- input is **Z**, going into a convolution

netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))

netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))

-- state size: **(ngf*8) x 4 x 4**

netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))

netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))

-- state size: **(ngf*4) x 8 x 8**

netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))

netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))

-- state size: **(ngf*2) x 16 x 16**

netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))

netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

-- state size: **(ngf) x 32 x 32**

netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))

netG:add(nn.Tanh())

-- state size: **(nc) x 64 x 64**

netG:apply(weights_init)

## netD

local netD = nn.Sequential()

-- input is **(nc) x 64 x 64**

netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))

netD:add(nn.LeakyReLU(0.2, true))

-- state size: **(ndf) x 32 x 32**

netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))

netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))

-- state size: **(ndf*2) x 16 x 16**

netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))

netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))

-- state size: **(ndf*4) x 8 x 8**

netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))

netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))

-- state size: **(ndf*8) x 4 x 4**

netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))

netD:add(nn.Sigmoid())

-- state size: **1 x 1 x 1**

netD:add(nn.View(1):setNumInputDims(3))

-- state size: **1**

netD:apply(weights_init)

---

## check netG and netD

**print(netG)**

nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> output]

  (1): nn.SpatialFullConvolution(100 -> 512, 4x4) without bias

  (2): nn.SpatialBatchNormalization (4D) (512)

  (3): nn.ReLU

  (4): nn.SpatialFullConvolution(512 -> 256, 4x4, 2,2, 1,1) without bias

  (5): nn.SpatialBatchNormalization (4D) (256)

  (6): nn.ReLU

  (7): nn.SpatialFullConvolution(256 -> 128, 4x4, 2,2, 1,1) without bias

  (8): nn.SpatialBatchNormalization (4D) (128)

  (9): nn.ReLU

  (10): nn.SpatialFullConvolution(128 -> 64, 4x4, 2,2, 1,1) without bias

  (11): nn.SpatialBatchNormalization (4D) (64)

  (12): nn.ReLU

  (13): nn.SpatialFullConvolution(64 -> 3, 4x4, 2,2, 1,1) without bias

  (14): nn.Tanh

}

**print(netD)**

nn.Sequential {

  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> output]

  (1): nn.SpatialConvolution(3 -> 64, 4x4, 2,2, 1,1) without bias

  (2): nn.LeakyReLU(0.2)

  (3): nn.SpatialConvolution(64 -> 128, 4x4, 2,2, 1,1) without bias

  (4): nn.SpatialBatchNormalization (4D) (128)

  (5): nn.LeakyReLU(0.2)

  (6): nn.SpatialConvolution(128 -> 256, 4x4, 2,2, 1,1) without bias

  (7): nn.SpatialBatchNormalization (4D) (256)

  (8): nn.LeakyReLU(0.2)

  (9): nn.SpatialConvolution(256 -> 512, 4x4, 2,2, 1,1) without bias

  (10): nn.SpatialBatchNormalization (4D) (512)

  (11): nn.LeakyReLU(0.2)

  (12): nn.SpatialConvolution(512 -> 1, 4x4) without bias

  (13): nn.Sigmoid

  (14): nn.View(1)

}

---

https://github.com/torch/nn/blob/master/doc/convolution.md#spatialfullconvolution

### SpatialFullConvolution

```
module = nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH], [adjW], [adjH])
```

Applies a 2D full convolution over an input image composed of several input planes. The `input` tensor in `forward(input)`is expected to be a 3D or 4D tensor. Note that instead of setting `adjW` and `adjH`, SpatialFullConvolution also accepts a table input with two tensors: `{convInput, sizeTensor}` where `convInput` is the standard input on which the full convolution is applied, and the size of `sizeTensor` is used to set the size of the output. Using the two-input version of forward will ignore the `adjW` and `adjH` values used to construct the module. The layer can be used without a bias by module:noBias().

Other frameworks call this operation "In-network Upsampling", "Fractionally-strided convolution", "Backwards Convolution," "Deconvolution", or "Upconvolution."

The parameters are the following:

- `nInputPlane`: The number of expected input planes in the image given into `forward()`.
- `nOutputPlane`: The number of output planes the convolution layer will produce.
- `kW`: The kernel width of the convolution
- `kH`: The kernel height of the convolution
- `dW`: The step of the convolution in the width dimension. Default is `1`.
- `dH`: The step of the convolution in the height dimension. Default is `1`.
- `padW`: Additional zeros added to the input plane data on both sides of width axis. Default is `0`. `(kW-1)/2` is often used here.
- `padH`: Additional zeros added to the input plane data on both sides of height axis. Default is `0`. `(kH-1)/2` is often used here.
- `adjW`: Extra width to add to the output image. Default is `0`. Cannot be greater than dW-1.
- `adjH`: Extra height to add to the output image. Default is `0`. Cannot be greater than dH-1.

If the input image is a 3D tensor `nInputPlane x height x width`, the output image size will be `nOutputPlane x oheight x owidth` where

```
owidth  = (width  - 1) * dW - 2*padW + kW + adjW
oheight = (height - 1) * dH - 2*padH + kH + adjH
```

Further information about the full convolution can be found in the following paper: [Fully Convolutional Networks for Semantic Segmentation](http://www.cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

---

https://github.com/torch/nn/blob/master/doc/convolution.md#spatialconvolution

### SpatialConvolution

```
module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
```

Applies a 2D convolution over an input image composed of several input planes. The `input` tensor in `forward(input)` is expected to be a 3D tensor (`nInputPlane x height x width`).

The parameters are the following:

- `nInputPlane`: The number of expected input planes in the image given into `forward()`.
- `nOutputPlane`: The number of output planes the convolution layer will produce.
- `kW`: The kernel width of the convolution
- `kH`: The kernel height of the convolution
- `dW`: The step of the convolution in the width dimension. Default is `1`.
- `dH`: The step of the convolution in the height dimension. Default is `1`.
- `padW`: Additional zeros added to the input plane data on both sides of width axis. Default is `0`. `(kW-1)/2` is often used here.
- `padH`: Additional zeros added to the input plane data on both sides of height axis. Default is `0`. `(kH-1)/2` is often used here.

Note that depending of the size of your kernel, several (of the last) columns or rows of the input image might be lost. It is up to the user to add proper padding in images.

If the input image is a 3D tensor `nInputPlane x height x width`, the output image size will be `nOutputPlane x oheight x owidth` where

```
owidth  = floor((width  + 2*padW - kW) / dW + 1)
oheight = floor((height + 2*padH - kH) / dH + 1)
```

The parameters of the convolution can be found in `self.weight` (Tensor of size `nOutputPlane x nInputPlane x kH x kW`) and `self.bias` (Tensor of size `nOutputPlane`). The corresponding gradients can be found in `self.gradWeight` and `self.gradBias`.

The output value of the layer can be precisely described as:

```
output[i][j][k] = bias[k]
  + sum_l sum_{s=1}^kW sum_{t=1}^kH weight[s][t][l][k]
                                    * input[dW*(i-1)+s)][dH*(j-1)+t][l]
```

---

