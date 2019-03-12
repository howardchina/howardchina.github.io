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

---

**Deep RNN**

*h', y = f<sub>1</sub>(h, x)  b', c = f<sub>2</sub>(b, y) ...*

* Concatenate several RNNs' input and output.

* ![1551157250482]({{site.url}}/static/img/posts/1551157250482.png)

---

**Bidirectional RNN**

*h', a = f<sub>1</sub>(h, x)  b', c = f<sub>2</sub>(b, x) y = f<sub>3</sub>(a, c)*

* input x from time 0 to time n-1
* input x from time n-1 to time 0
* output y sequencially

* ![1551158909175]({{site.url}}/static/img/posts/1551158909175.png)

---

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

---

**LSTM**

* short time memory: h changes fast -> h<sup>t-1</sup> and h<sup>t</sup> can be very different;

* long time memory: c changes slow -> c<sup>t-1</sup> and c<sup>t</sup> can be very similar.

* ![1551159566140]({{site.url}}/static/img/posts/1551159566140.png)

---

* z
* Input gate z<sup>i</sup>
* forget gate z<sup>f</sup>
* output gate z<sup>o</sup>
* ![1551924581619]({{site.url}}/static/img/posts/1551924581619.png)

---

* **Non-linear transform and activation function** is represented by **thick arrow**;
* ![1551419813143]({{site.url}}/static/img/posts/1551419813143.png)

---

* **different colors** represent **different transforms**;
* thin arrow represent the ordinary linear data flow;
* dash arrow represents a duplicate;
* a **"peephole"** helps input c<sup>t-1</sup> and **multiply c<sup>t-1</sup> by a diagonal matrix**.
* multiply vector **W** by vector **(x<sup>t</sup> h<sup>t-1</sup> c<sup>t-1</sup>)** , in which **c<sup>t-1</sup>** is multiplied by a diagonal matrix.
* ![1551161085709]({{site.url}}/static/img/posts/1551161085709.png)

---

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
* LSTM can be unfolded when it is transmitted forward.

---

**GRU**

* [zhihu](https://zhuanlan.zhihu.com/p/32481747)

* h' is a hidden state between -1~1 coming from a tanh function which transformed x<sup>t</sup> and h<sup>t-1'</sup> ;
* h<sup>t-1'</sup> is come from r and h<sup>t-1</sup>;
* r controls the input gate such as the z<sup>i</sup> at LSTM;
* z is similar to forget gate;
* 1-z is similar to output gate;
* ![1552321894501]({{site.url}}/static/img/posts/1552321894501.png)

---

## Sequence Generation

**Generation**

* x: is the token generated at the last time step and represented by 1-of-N encoding;
* y: is distribution over the token, which generates a token by sampling. 
* ![1552358724167]({{site.url}}/static/img/posts/1552358724167.png)

---

 **Generation**

* y: P(token \| conditions), a character or word, is generated at each time by RNN;
* started by **BOS** (Begin Of Sentence);
* ended with **EOS** (End Of Sentence);
* sample rather than argmax so as to generate different sentence.
* ![1551429195709]({{site.url}}/static/img/posts/1551429195709.png)

---

* minimizing cross-entropy
* ![1552360096928]({{site.url}}/static/img/posts/1552360096928.png)

---

**Naive Pixel-RNN**

* To consider colors as tokens;
* pixels are generated in a zig-zag sequence that against our visual intuition.
* ![1552360313381]({{site.url}}/static/img/posts/1552360313381.png)

---

**Improved Pixel-RNN**

* choose a more reasonable direction for sequence generation.
* ![1552360575675]({{site.url}}/static/img/posts/1552360575675.png)

---

## Conditional Sequence Generation

**Conditional Generation**

* Generate sentences based on conditions, rather than some random sentences;
* such as **Caption Generation** generating specific captions for given images or **Chat-bot** replying given questions by generated answers.

---

**Image Caption Generation**

* Represent the input condition as a vector, and consider the vector as the input of RNN generator.
* ![1552375760163]({{site.url}}/static/img/posts/1552375760163.png)

---

**Sequence-to-sequence learning**

* such as Machine translation / Chat-bot;

* **Encoder** generates **a compressed block** at the last step containing the information of the whole sentences;
* **Decoder** generates a series of tokens sequentially and repeatedly from this block.
* Encoder and decoder are jointly training.
* ![1552377363435]({{site.url}}/static/img/posts/1552377363435.png)

---

**Conditional Generation**

* Need to consider longer context during chatting;
* ![1552378054764]({{site.url}}/static/img/posts/1552378054764.png)

---

**Dynamic Conditional Generation**

* Encoder generates a series of hidden tokens which is similar to a "hidden database";
* Decoder extracts tokens from the "hidden database", and rearranges these tokens to a reasonable sequence before input.
* ![1552378465721]({{site.url}}/static/img/posts/1552378465721.png)

---

**Attention-based model**

* **z<sup>0</sup>** is the hidden variable of the decoder;
* **match** is a small NN whose input is **z** and **h**, and outputs a scalar **&alpha;**;
* &alpha; = h<sup>T</sup> W z
* ![1552397407052]({{site.url}}/static/img/posts/1552397407052.png)

---

**Attention-based model**

* &alpha;_hats are generated by a softmax function whose input are &alpha;s;
* the sum of &alpha;_hats is equal to **1**;
* **c<sup>0</sup>** = &Sigma; &alpha;_hat<sub>0</sub><sup>i</sup> h<sup>i</sup>
* c<sup>0</sup> and z<sup>0</sup> update the hidden status **z<sup>1</sup>** of decoder and generate the first token such as "machine" in this example.
* ![1552399162199]({{site.url}}/static/img/posts/1552399162199.png)

---

**Attention-based model**

* repeatedly match the new z<sup>1</sup>, and calculate &alpha; and c<sup>1</sup> until the decoder generate \<EOS\>;
* jointly training
* ![1552399687209]({{site.url}}/static/img/posts/1552399687209.png)
* ![1552399706661]({{site.url}}/static/img/posts/1552399706661.png)
* ![1552399722807]({{site.url}}/static/img/posts/1552399722807.png)

---

**Image Caption Generation**

* CNN plays a role of encoder who generate a vector for each region;
* then match z<sup>0</sup> to each vector, as is usually done with attention-based model;
* ![1552400894062]({{site.url}}/static/img/posts/1552400894062.png)
* ![1552400918393]({{site.url}}/static/img/posts/1552400918393.png)
* ![1552400946906]({{site.url}}/static/img/posts/1552400946906.png)

---

**Effects of Image Caption Generation** 

* Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio, “Show, Attend and Tell: Neural Image Caption Generation with Visual Attention”, ICML, 2015

* ![1552401181911]({{site.url}}/static/img/posts/1552401181911.png)
* ![1552401216176]({{site.url}}/static/img/posts/1552401216176.png)
* Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, Aaron Courville, “Describing Videos by Exploiting Temporal Structure”, ICCV, 2015
* ![1552401320206]({{site.url}}/static/img/posts/1552401320206.png)

---

## Tips for Generation

**Attention Weight**

* The regularization term helps attention evenly distributed on each frame;
* for each component  or frame, the sum of attention is close to &tau;;
* ![1552402322349]({{site.url}}/static/img/posts/1552402322349.png)

---

**Mismatch between Train and Test**

* Exposure Bias
* 