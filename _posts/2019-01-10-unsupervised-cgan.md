---
layout: post
title:  Unsupervised conditional GAN
date:   2019-01-10 22:24:00 +0800
categories: [gan]
---

2018 Lecture 3

## Cycle GAN

related techs:

* style transformation
* Direct transformation
* Autoencoder-based domain transformation

Notice:

* Simpler generator makes the input and output more closely related.
* Deep network needs constrain to make the input and output consistent.

Related GANs:

* Disco GAN
* Dual GAN

### StarGAN

multiple domain

![1547131257792]({{site.url}}/static/img/posts/1547131257792.png)

### Project to Common Space

* Share hidden layer.
* Domain Discriminator
* Cycle Consitency
* Semantic Consistency

![1547131481544]({{site.url}}/static/img/posts/1547131481544.png)

![1547131572879]({{site.url}}/static/img/posts/1547131572879.png)

![1547131690959]({{site.url}}/static/img/posts/1547131690959.png)

![1547131796723]({{site.url}}/static/img/posts/1547131796723.png)

### Reference
* Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros, Unpaired Image-toImage Translation using Cycle-Consistent Adversarial Networks, ICCV, 2017
* Zili Yi, Hao Zhang, Ping Tan, Minglun Gong, DualGAN: Unsupervised Dual
Learning for Image-to-Image Translation, ICCV, 2017
* Tomer Galanti, Lior Wolf, Sagie Benaim, The Role of Minimal Complexity Functions in Unsupervised Learning of Semantic Mappings, ICLR, 2018
* Yaniv Taigman, Adam Polyak, Lior Wolf, Unsupervised Cross-Domain Image Generation, ICLR, 2017
* Asha Anoosheh, Eirikur Agustsson, Radu Timofte, Luc Van Gool, ComboGAN: Unrestrained Scalability for Image Domain Translation, arXiv, 2017
* Amélie Royer, Konstantinos Bousmalis, Stephan Gouws, Fred Bertsch, Inbar Mosseri, Forrester Cole, Kevin Murphy, XGAN: Unsupervised Image-to-Image Translation for Many-to-Many Mappings, arXiv, 2017
* Guillaume Lample, Neil Zeghidour, Nicolas Usunier, Antoine Bordes, Ludovic Denoyer, Marc'Aurelio Ranzato, Fader Networks: Manipulating Images by Sliding Attributes, NIPS, 2017
* Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, Jiwon Kim, Learning to Discover Cross-Domain Relations with Generative Adversarial Networks, ICML, 2017
* Ming-Yu Liu, Oncel Tuzel, “Coupled Generative Adversarial Networks”, NIPS, 2016
* Ming-Yu Liu, Thomas Breuel, Jan Kautz, Unsupervised Image-to-Image Translation Networks, NIPS, 2017
* Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo, StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation, arXiv, 2017