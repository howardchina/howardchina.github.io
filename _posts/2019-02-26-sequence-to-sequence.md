---
layout: post
title:  Sequence Generation
date:   2019-02-26 00:00:00 +0800
categories: [GAN]
---

Outline

* RNN with Gated Mechanism
* Sequence Generation
* Conditional Sequence Generation
* Tips for Generation

---

## RNN with Gated Mechanism

**Recurrent Neural Network**

Given function f: *h', y=f(h,x)*

* note: **h**: last hidden input, **h'**: current hidden output, **x**: input, **y**: output

* No matter how long the input/output sequence is, we only need one function f.

* ![1551157215565]({{site.url}}/static/img/posts/1551157215565.png)

**Deep RNN**

*h', y = f<sub>1</sub>(h, x)  b', c = f<sub>2</sub>(b, y) ...*

* Concatenate several RNNs' input and output.

* ![1551157250482]({{site.url}}/static/img/posts/1551157250482.png)

**Bidirectional RNN**

*h', a = f<sub>1</sub>(h, x)  b', c = f<sub>2</sub>(b, x) y = f<sub>3</sub>(a, c)*

* input x from time 0 to time n-1
* input x from time n-1 to time 0
* output y sequencially

* ![1551158909175]({{site.url}}/static/img/posts/1551158909175.png)

**Naive RNN**

Given function f: *h', y = f(h, x)*

* W is a matrix
* h is a vector
* h has the same dimension with h'
* h' is transformed from h and x
* y is transformed from h'
* notice: superscript h at W<sup>h</sup> means this parameter W correspond to hidden layer h. So does W<sup>i</sup> and W<sup>o</sup> to input layer and output layer.
* ps: sigmoid is better performed than ReLU in RNN.
* ![1551159147019]({{site.url}}/static/img/posts/1551159147019.png)

**LSTM**

* short time memory: h changes fast -> h<sup>t-1</sup> and h<sup>t</sup> can be very different;
* long time memory: c changes slow -> c<sup>t-1</sup> and c<sup>t</sup> can be very similar.
* ![1551159566140]({{site.url}}/static/img/posts/1551159566140.png)
* z
* Input gate z<sup>i</sup>
* forget gate z<sup>f</sup>
* output gate z<sup>o</sup>
* ![1551924581619]({{site.url}}/static/img/posts/1551924581619.png)
* **Non-linear transform and activation function** is represented by **thick arrow**;
* ![1551419813143]({{site.url}}/static/img/posts/1551419813143.png)
* **different colors** represent **different transforms**;
* thin arrow represent the ordinary linear data flow;
* dash arrow represents a duplicate;
* a **"peephole"** helps input c<sup>t-1</sup> and **multiply c<sup>t-1</sup> by a diagonal matrix**.
* multiply vector **W** by vector **(x<sup>t</sup> h<sup>t-1</sup> c<sup>t-1</sup>)** , in which **c<sup>t-1</sup>** is multiplied by a diagonal matrix.
* ![1551161085709]({{site.url}}/static/img/posts/1551161085709.png)
* about dimension:
  * c, h, x and z are of the same dimension;
  * dot product and matrix addition wouldn't change dimension.
  * matrix product transform [c, h, x] into z (which is of the same dimension to c, h and x).
* [z<sup>i</sup> * z ] input gate;
* [z<sup>f</sup> * c] forget gate;
* [z<sup>o</sup> * c'] output gate;
* the long memory, c', is updated by c and input, h and x, and obtained by 2 gate controllers, the forget gate and input gate;
* the short memory, h', is updated by c';
* output y' is transformed by an activation function input *W'* and *h'*.
* ![1551160946442]({{site.url}}/static/img/posts/1551160946442.png)

 **Sequence Generation**

![1551429195709]({{site.url}}/static/img/posts/1551429195709.png)

1. y<sup>1</sup>, y<sup>2</sup> and y<sup>3</sup> are conditional probability;
2. started by **BOS** (Begin Of Sentence);
3. ended with **EOS** (End Of Sentence);
4. sample rather than argmax so as to generate different sentence.

---

### Dynamic Conditional Generation

1. Encoder
   * to encode Chinese into Machine Language;
   * save these Machine Language in a "hidden database".
2. Decoder
   * to search and extract specific information in a form of c<sup>i</sup> from "hidden database";
   * transform such information c<sup>i</sup> into English.

---

### Attention-based model

key: z<sup>0</sup>

* match the key (z<sup>0</sup>) with each h<sup>i</sup>, output inner product;
* match might be a small NN whose input is h and z, output a scalar;


=======
Training

1. Minimize **cross-entropy** between each y<sup>t</sup> and training data;
2. Started with **BOS**;

![1551446739714]({{site.url}}/static/img/posts/1551446739714.png)

---

To consider image as sentence. But the relevance between adjacent pixels are not consisted with intuition.

![1551448266510]({{site.url}}/static/img/posts/1551448266510.png)

---

Pixel-RNN generate each pixel 

![1551450585640]({{site.url}}/static/img/posts/1551450585640.png)

