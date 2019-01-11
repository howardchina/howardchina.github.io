---
layout: post
title:  Feature extraction, InfoGAN, VAE-GAN, BiGAN
date:   2019-01-11 10:00:00 +0800
categories: [gan]
---

ref: GAN Lecture 7 (2018): Info GAN, VAE-GAN, BiGAN

### InfoGAN

* **Generator** put the input c (and z') into generated x.
  * c must have clear influence on x
  * **encoder**
* **Classifier** predict the code c that generates x
  * the classifier can recover c from x
  * **decoder**
* Classifier and Discriminator are parameter sharing

![1547219738894]({{site.url}}/static/img/posts/1547219738894.png)

![1547219789215]({{site.url}}/static/img/posts/1547219789215.png)

### VAE-GAN

* encoder
  * Minimize **reconstruction error**
  * **z** close to normal
* decoder (Generator)
  * minimize **reconstruction error**
  * cheat discriminator
* discriminator
  * discriminate real, generated and reconstructed images.

![1547219894341]({{site.url}}/static/img/posts/1547219894341.png)

![1547219927532]({{site.url}}/static/img/posts/1547219927532.png)

### BiGAN

Quiet different from VAE-GAN.

* give the **pair** of **input of encoder x<sup>i</sup> and output of decoder z~<sup>i</sup>** high score

* give the **pair** of **output of decoder x~<sup>i</sup> and input of decoder z<sup>i</sup>**  high score
* **encoder and decoder work together**, and try to fool the discriminator.
* This autoencoder will learn the semantic  information of disctribution.

![1547220894005]({{site.url}}/static/img/posts/1547220894005.png)

### Triple GAN

* Semi-supervised

![1547221428977]({{site.url}}/static/img/posts/1547221428977.png)

### Domain-adversarial training

3 networks

* feature extractor (generator)
* domain classifier (discriminator)
* label predictor
* iteratively training is more stable.

![1547221714468]({{site.url}}/static/img/posts/1547221714468.png)

### Feature Disentangle

inspired by domain-adversarial training

![1547222127478]({{site.url}}/static/img/posts/1547222127478.png)

* input different speakers into a **speaker encoder (generator)**
* distance between two speakers is larger than a **threshold**.

![1547222145377]({{site.url}}/static/img/posts/1547222145377.png)

* for a **speaker classifier (discriminator) with two phonetic input**, same speaker get high score, different speakers get low score.

![1547222164866]({{site.url}}/static/img/posts/1547222164866.png)

![1547222179803]({{site.url}}/static/img/posts/1547222179803.png)