---
layout: post
title:  Computational Graph
date:   2019-02-18 00:00:00 +0800
categories: [GAN]
---

**Computational Graph** 

* A "language" describing a function
  * **Node**: variable
  * **Edge**: operation

*Example* 

![1550479087574]({{site.url}}/static/img/posts/1550479087574.png)

**Chain Rule** for divergence

![1550479361369]({{site.url}}/static/img/posts/1550479361369.png)

***Calculate the result** of a graph or an equation, such as*: e=(x+b)\*(b+1):

1. To calculate the divergence at every each edge;
2. To substitute the value (a=3, b=2) into equation

***Calculate divergence** on graph, such as:* &delta;e/&delta;b=8 and &delta;e/&delta;a=3:

1. Forward mode is similar to DFS;
   * ![1550707932183]({{site.url}}/static/img/posts/1550707932183.png)
   * ![1550707962643]({{site.url}}/static/img/posts/1550707962643.png)
2. Reverse mode is similar to BFS.
   * ![1550480632035]({{site.url}}/static/img/posts/1550480632035.png)

**Parameter sharing**:  the same parameters appearing in different nodes such as *y=xe<sup>x<sup>2</sup></sup>*

![1550712646589]({{site.url}}/static/img/posts/1550712646589.png)

* To compute each x<sub>i</sub> divergence separately and summarize x<sub>i</sub> to x together;
* three  x<sub>i</sub> are in equation above but substituted by a, b and c.

Loss Function of **Feedforward**

![1550806569053]({{site.url}}/static/img/posts/1550806569053.png)

**Back-propagation on graph**

![1550805395640]({{site.url}}/static/img/posts/1550805395640.png)

**Jacobian Matrix** for calculating the partial derivate on th edge such as *&delta;a<sup>1</sup>/&delta;z<sup>1</sup>=?*, in which both *a<sup>1</sup> and z<sup>1</sup>* are vector:

![1550805304204]({{site.url}}/static/img/posts/1550805304204.png)

1. To calculate **&delta;C/&delta;y**. **y_hat** tells the **position** **r**. **y** tells the **value y<sub>r</sub>**.

   ![1550807330918]({{site.url}}/static/img/posts/1550807330918.png)

   ​	y<sub>r</sub> close to 1. C close to 0.

2. To calculate **&delta;y/&delta;z<sup>2</sup>**. Activation function **&sigma;** is **sigmoid** function. Then **&delta;y/&delta;z<sup>2</sup>** is a Jacobian matrix and a **diagonal** matrix.

   ![1550812912967]({{site.url}}/static/img/posts/1550812912967.png)

3. To calculate **&delta;z<sup>2</sup>/&delta;a<sup>1</sup>**, a Jacobian matrix, whose i-th row and j-th column is **&delta;z<sup>2</sup><sub>i</sub>/&delta;a<sup>1</sup><sub>j</sub>=w<sub>ij</sub><sup>2</sup>**. So the answer is W<sup>2</sup>.

   ​	Because:

   ![1550821086259]({{site.url}}/static/img/posts/1550821086259.png)

   ​	As for z<sub>i</sub><sup>2</sup>:

   ![1550821049998]({{site.url}}/static/img/posts/1550821049998.png)

4. To calculate **&delta;z<sup>2</sup>/&delta;W<sup>2</sup>**, a tensor with 3 dimensions. Considering W<sup>2</sup> as a **mxn** vector.

   ![1550923551221]({{site.url}}/static/img/posts/1550923551221.png)