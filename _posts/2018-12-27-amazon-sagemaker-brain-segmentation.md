---
layout: post
title:  mxnet brain segmentation
date:   2018-12-27 11:56:00 +0800
categories: [brain]
---


the goal of semantic segmentation is to make classifications on an image at the pixel-level, producing a classification “mask.”

### CNN Architectures

We will train two networks, U-Net and ENet.

* **ENet**: Introduced in the paper [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147), ENet is designed to be low-latency and to operate in environments with low compute capacity (e.g. edge devices). Compared to existing architectures, ENet optimizes for processing time over accuracy.

### Dataset

In this notebook, we’ll be using Brain MRI data from the [Open Access Series of Imaging Studies (OASIS)](http://www.oasis-brains.org/). This project offers a wealth of neuroimaging datasets; we’ll be looking at a small subset of cross-sectional brain MRIs.

**Note:** You need to request access on the OASIS site to get the data. In this tutorial, I’ll be using the `disc1.tar.gz`file from the [OASIS-1](http://www.oasis-brains.org/#data) data set.