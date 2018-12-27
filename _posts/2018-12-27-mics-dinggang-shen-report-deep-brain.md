---
layout: post
title:  Deep Learning in Brain Quantification and Cancer Radiotherapy
date:   2018-12-27 11:12:00 +0800
categories: [brain]

---

Dinggang Shen

University of North Carolina at Chapel Hill

47 Pages

### Outline

- Brain (Infants) **Segmentation & Registration**
- Disease Diagnosis (Structural MRI, Functional MRI) **Classification**
- Radiotherapy (MRI->CT, PET Attenuation Correction) **Image Synthesis & Organ Delineation**

### Overall Goal

- Baby Connectome Project (BCP): Autism? [6 M, 3~4 Y]
- Alzheimer's Disease Neuroimaging Initiative (ADNI): Alzheimer's Disease (AD) [70 Y (MCI) , 75 Y]

### NIH Lifespan Human Connectome Projects

### Brain Functional Maturation

### The Human Brain in the First Year of Life

**Challenges:** Low tissue contrast (especially ~6 months of age) and low spatial resolution

### Segmentation

* Tissue Segmentation

  * input: T1w, T2w, Anotomy

  * architecture: UNet, Dense Block

  * output: CSF, GM, WM

    Li Wang et al., HBM 2018; MICCAI 2018

* Surface Parcellation

  * architecture: Spaherical Patch-based CNN with graph cuts

    Z. Wu, G. Li, L. Wang, F. Shi, W. Lin, J.H. Gilmore, D. Shen, “Registration-free infant cortical surface parcellation using spherical patchwise deep convolutional neural networks”, MICCAI 2018, Granada, Spain, Sep. 16-20, 2018.

### Registration

* Learning-based Registration

  J. Fan, X. Cao, Z. Xue, P. Yap, D. Shen, “Adversarial Similarity Network for Evaluating Image Alignment in Deep Learning based Registration”, MICCAI 2018, Granada, Spain, Sep. 16-20, 2018.

### Disease Diagnosis

* Anatomical Landmark based Deep Learning using Structural MRI

  M. Liu, J. Zhang, E. Adeli, D. Shen, “Landmark-based Deep Multi-Instance Learning for Brain Disease Diagnosis,” Medical Image Analysis, 2018.

### Radiotherapy

* Image Synthesis

  * Deep Learning-based Automatic Synthesis of CT from MRI

    T. Huynh, Y. Gao, J. Kang, L. Wang, P. Zhang, J. Lian, D. Shen, “Estimating CT Image from MRI Data Using Structured Random Forest
    and Auto-context Model”, IEEE Transactions on Medical Imaging, 2016.

    D. Nie∗, R. Trullo∗, J. Lian, C. Petitjean, S. Ruan, D. Shen, “Medical Image Synthesis with Context-Aware Generative Adversarial
    Networks”, MICCAI, 2017.

* Organ Delineation

  * Attention-based Semi-supervised Networks for Prostate Segmentation
  * 