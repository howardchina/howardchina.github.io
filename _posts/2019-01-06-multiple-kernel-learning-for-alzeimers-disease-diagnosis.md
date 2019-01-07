---
layout: post
title:  PR Structured sparsity regularized multiple kernel learning for Alzheimers disease diagnosis
date:   2019-01-06 23:14:00 +0800
categories: [brain]
---

Keyword: Multi-modality, sparsity

Jialin Peng 1, Dinggang Shen 2

1. College of Computer Science and Technology, Huaqiao University, Xiamen, China
2. Department of Brain and Cognitive Engineering, Korea University, Seoul, Korea

**Abstract:**

Multimodal data fusion

* imaging (phenotype)
* genetic (genotype)

Structured sparsity regularized multiple kernel learning method. Represent each feature with a distinct kernel as a basis, followed by grouping the kernels according to modalities. 

The proposed regularizer enforced on kernel weights is 

* to sparsely select concise feature set within each homogeneous group
* and fuse the heterogeneous feature groups by taking advantage of dense norms

Evaluation on Alzheimer's Disease Neuroimaging Initiative (ADNI) database.

discover brain regions and SNPs relevant to AD.

### 1.Introduction

leverage both phenotype and genotype information, e.g., MRI, PET, and SNPs 

![1546790517910]({{site.url}}/static/img/posts/1546790517910.png)

![1546790467339]({{site.url}}/static/img/posts/1546790467339.png)