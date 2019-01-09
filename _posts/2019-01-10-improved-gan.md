---
layout: post
title:  Improved GAN
date:   2019-01-10 00:01:00 +0800
categories: [gan]
---

![1547050170276]({{site.url}}/static/img/posts/1547050170276.png)

### f-divergence

P (data) and Q (generator) are two distributions. p(x) and q(x) are the probability of sampling x.

![1547051038643]({{site.url}}/static/img/posts/1547051038643.png)

if *f* is convex, D<sub>*f*</sub> (P\|\|Q)has the smallest value, which is 0.

### Fenchel Conjugate

Every convex function *f* has a conjugate function *f* <sup>*</sup>.

**(f <sup>\*</sup>) <sup>\*</sup> = f**

f(x)=xlogx   <------>   f<sup>*</sup>(t) = exp(t-1)

![1547051462525]({{site.url}}/static/img/posts/1547051462525.png)

### (f-divergence's) Connection with GAN

D is a function whose input is x, and output is t.

![1547054190120]({{site.url}}/static/img/posts/1547054190120.png)

![1547054271565]({{site.url}}/static/img/posts/1547054271565.png)

![1547054408877]({{site.url}}/static/img/posts/1547054408877.png)

### Double-loop v.s. Single-step