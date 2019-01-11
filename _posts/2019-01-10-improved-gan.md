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

Original paper of GAN: double-loop algorithm

* in each iteration
  * Given a generator
    * update the parameters many times to find D
  * Update generator one

Paper of f-GAN: Single-stop algorithm

* in each iteration, given G and D at time t
  * update D and G in one backpropogation

## WGAN

**Earth Mover's Distance**

2-d: W(P,Q)=d

More complex distribution: many possible "moving plans".

A "moving plan" is a matrix:

![1547102779260]({{site.url}}/static/img/posts/1547102779260.png)

**Lipschitz Function**

Output change smaller than Input change.

K=1 for "1-Lipschitz". **Do not change fast.**

\|\|D(x1)-D(x2)\|\|<=\|\|x1-x2\|\|

![1547103699924]({{site.url}}/static/img/posts/1547103699924.png)

**Gradient descent**

Weight clipping: force the  weights w between c and -c.

**Algorithm of WGAN**

* No sigmoid for the output.
* no log
* weight cliping

Good CNN generator structure: Number of filters at each layer is **two times** than that at previous layer.

**Wasserstein distance can estimate the quality of generated image.**

### Gradient Penalty

![1547104911937]({{site.url}}/static/img/posts/1547104911937.png)

How to sample P<sub>penalty</sub>?

### Sentence Generation

### Transformation

* Paired data

* Unpaired data

  * Cycle GAN, Disco GAN

![1547120584225]({{site.url}}/static/img/posts/1547120584225.png)

![1547120654090]({{site.url}}/static/img/posts/1547120654090.png)

That's it.