---
layout: post
title:  Sequence Generation
date:   2019-02-26 00:00:00 +0800
categories: [GAN]
---

**Recurrent Neural Network**

​	Given function f: *h', y=f(h,x)*

​	No matter how long the input/output sequence is, we only need one function f.

  ![1551157215565]({{site.url}}/static/img/posts/1551157215565.png)

---

**Deep RNN**

​	*h', y = f<sub>1</sub>(h, x)  b', c = f<sub>2</sub>(b, y) ...*

​	![1551157250482]({{site.url}}/static/img/posts/1551157250482.png)

---

**Bidirectional RNN**

​	*h', a = f<sub>1</sub>(h, x)  b', c = f<sub>2</sub>(b, x) y = f<sub>3</sub>(a, c)*

​	![1551158909175]({{site.url}}/static/img/posts/1551158909175.png)

---

**Naive RNN**

​	Given function f: *h', y = f(h, x)*

​	![1551159147019]({{site.url}}/static/img/posts/1551159147019.png)

​	notice: superscript h at W<sup>h</sup> means this parameter W correspond to hidden layer h. So does W<sup>i</sup> and W<sup>o</sup> to input layer and output layer.

​	ps: sigmoid is better perform than ReLU in RNN.

---

**LSTM**

​	![1551159566140]({{site.url}}/static/img/posts/1551159566140.png)

​	h changes fast -> h<sup>t-1</sup> and h<sup>t</sup> can be very different

​	c changes slow -> c<sup>t-1</sup> and c<sup>t</sup> can be very similar 

---

​	![1551159974080]({{site.url}}/static/img/posts/1551159974080.png)

1. z
2. Input gate z<sup>i</sup>
3. forget gate z<sup>f</sup>
4. output gate z<sup>o</sup>

---

​	![1551161085709]({{site.url}}/static/img/posts/1551161085709.png)

vector **W** multiply by vector **(x<sup>t</sup> h<sup>t-1</sup> c<sup>t-1</sup>)** , in which **c<sup>t-1</sup>** multiply by a diagonal matrix.

​	![1551160946442]({{site.url}}/static/img/posts/1551160946442.png)

1. input gate decide to drop input z or not.
2. forget gate decide to drop old c<sup>t-1 </sup>or not.
3. output gate decide to drop new c<sup>t</sup> from output or not
4. c<sup>t</sup> is updated by c<sup>t-1</sup> (the old long memory), z<sup>f</sup>, z (the current input) and z<sup>i</sup> and even does not have a non-linear function in this step.
5. h<sup>t</sup> (the short memory) is updated by c<sup>t</sup> (the new long memory), the activation function and the output gate.
6. y<sup>t</sup> (the current output) is an activation function consisting of *W'* and *h<sup>t </sup>*(as parameters).

---

