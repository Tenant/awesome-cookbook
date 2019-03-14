# Deep Learning

[CS294-112 Fall 2017]([CS294-112 Fall 2017](https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3))

[Berkeley-Deep Course](http://rll.berkeley.edu/deeprlcourse/)

[Past Class Resource](<http://rail.eecs.berkeley.edu/deeprlcoursesp17/index.html>)

## 1. Neural Network and Deep Learning

### 1.1 What is a Neural Network

| Type of Neural Network       | Application                             |
| :--------------------------- | :-------------------------------------- |
| Standard Neural Network      | Website advertising                     |
| Convolutional Neural Network | Photo tagging, Autonomous driving       |
| Recurrent Neural Network     | Speech recognition, machine translation |

| Type of Data      | Features                                      |
| ----------------- | --------------------------------------------- |
| Structured data   | You know exactly the meaning of each element. |
| Unstructured data | Like audio, image, and even text.             |

### 1.2 Why Is Deep Learning Taking Off

Scale drives Deep Learning Process: if you need a high performance network, the first thing you need is training a big enough neural network, and second one is you need a lot of data. So, Data and Computation matters, and a good  algorithm design may improve Computation performance massively.

If you only have a small training datasets, SVM or logistic regression may perform better than NN, otherwise in the big data regime NN does better.

**The circle of designing a Neural Network**: Idea -> Code -> Experiment -> Idea

| Activating functions | Features           |
| -------------------- | ------------------ |
| Sigmoid              | Convergent slowly. |
| ReLU                 | Convergent fast.   |

### 1.3 Binary Classification

==Cost function of Logistic Regression==:

$$\begin{align}J(w,b)=\frac{1}{m}\sum_{i=1}^m -(y^{(i)}\log {\hat y}^{(i)} + (1-y)\log (1-{\hat y}))\end{align}$$

==Simple Logistic Regression==

$$\begin{align}J=0, dw_1=0,dw_2=0,db=0\end{align}$$

$$\begin{align}i=1:m\end{align}$$

## 2. Improving Deep Neural Network

### 2.1 Train/Dev/Test Datasets

| Magnitude of Dataset | Train   | Dev        | Test       |
| -------------------- | ------- | ---------- | ---------- |
| Normal Dataset       | 0.6%    | 0.2%       | 0.2%       |
| Big Dataset          | 800,000 | max.10,000 | max.10,000 |

### 2.2 Basic Recipe for Machine Learning

Observe whether you NN fit the training data set well, and tell whether it's high bias? If the answer is yeas, try bigger network, training longer. If the answer is no, observe whether you NN fit the dev data set well,  and tell whether it's high variance? If the answer is yes, try more data, or try Regularization, otherwise you well be done.

==Deep learning can deal with high bias/variance without hurting the other one.==

### 2.3 Regularization

#### 2.3.1 L1 Regularization

$$\begin{align}\min_{w,b}J(w,b)=\frac{1}{m}\sum_{i=1}^m\ell({\hat y}^{(i)},y^{(i)}) +\frac{\lambda}{2m}|w|_1\end{align}$$

where $w\in R^n, b\in R$.

#### 2.3.2 L2 Regularization

$$\begin{align}\min_{w,b}J(w,b)=\frac{1}{m}\sum_{i=1}^m\ell({\hat y}^{(i)},y^{(i)}) +\frac{\lambda}{2m}||w||^2_2\end{align}$$

where $||w||^2_2=\sum_{j=1}^n w_j^2=w^Tw, w\in R^n, b\in R$.

#### 2.3.3 Neural Network L2 Regularization(Weights Delay Regularization)

$$\begin{align}\min_{w,b} J(w^{[1]},b^{[1]},...,w^{[L]},b^{[L]})=\frac{1}{m}\sum_{i=1}^m \ell({\hat y}^{(i)},y^{(i)})+\frac{\lambda}{2m}\sum_{\ell=1}^L ||w^{[\ell]}||^2\end{align}$$

where $||w^{[\ell]}=\sum_{i=1}^{n^{[\ell]}} \sum_{j=1}^{n^{[\ell-1]}} (w_{ij}^{[\ell]})^2$, $w^{[\ell]}\in R^{n^{[\ell]}\times n^{[\ell]-1}}$.

#### 2.3.4 Why Regularization Reduces Overfitting

If the regularization term $\lambda$ is very large, the parameters $W$ will be small, and as a consequence $Z$ will be small and the activation functions will work only in the linear area.

#### 2.3.5 Dropout Regularization

**Pseudo-code**:

```python
set keep_prop = 0.8
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prop
a3 = np.multiply(a3, d3)
a3 /= keep_prob
```

**Intuition**: never relay on any single feature, so have to spread out weights. If the layer there are more nodes, you should set the ***keep_prop**** more lower, visa verse. You can implement Dropout Regularization on ==input layer==, but usually we don’t do it. Dropout Regularization is commonly used in ==Computer Vision==.

**Drawback**: Dropout regularization makes the cost function ill defined, so you need to relay on cost-iteration curve to judge your algorithm performance.

#### 2.3.6 Other Regularization Methods

| Methods           | ——                                                           |
| ----------------- | ------------------------------------------------------------ |
| Data Augmentation | Flip images, rotate images, zooming images, distort images, etc. |
| Early Stopping    | Plotting your training-dataset error and dev-dataset error. Early stopping may lead to a incompletely optimized training result compared to L2 regularization, but it is simple and doesn’t need to try the suitable value for $\lambda$ in L2 Norm. |

### 2.4 Normalizing Inputs

Normalizing your training dataset and dev dataset using the same parameters.

Normalizing inputs helps to converge faster.

==So do normalization anyway.==

### 2.5 Vanishing/Exploding Gradients

Set $g(z)=z$, and $b^{[i]}=0$, then $\hat y=W^{[L]}W^{[L-1]}...W^{[1]}x$.

If we set $W^{[\ell]}$= $$\left (\begin{matrix}1.5 & 0 \\ 0 & 1.5\end{matrix}\right)$$, $\hat y$ will be exponentially large;

if we set $W^{[\ell]}$=$$\left(\begin{matrix}0.5 & 0\\ 0 & 0.5\end{matrix}\right)$$, $\hat y$ will be exponentially small.

==By reasonably initializing the weights, we can partially solve this problem.==

### 2.6 Weights Initialization in a Deep Network

**ReLU**: Set $W^{[\ell]}$=np.random.randn($n^{[\ell]}$,$n^{[\ell]-1}$)$\times \sqrt{\frac{2}{n^{[\ell]-1}}}$.

**tanh**: Set $W^{[\ell]}=$np.random.randn($n^{[\ell]}$,$n^{[\ell]-1}$)$\times \tanh\sqrt{\frac{1}{n^{[\ell]-1}}}$.

### 2.7 Gradient Checking

$$\begin{align}f'(\theta)=\frac{f(\theta+\tau)-f(\theta-\tau)}{2\tau}\end{align}$$

Take $W^{[1]}$, $b^{[1]}$,…, $W^{[L]}, b^{[L]}$ and reshape them into a big vector $\theta$.

Take $dW^{[1]}$, $db^{[1]}$,…, $dW^{[L]}$, $db^{[L]}$ and reshape them into a big vector $d\theta$.

Check whether $d\theta_{approx}^{[i]}=\frac{J(\theta_1,...,\theta_i+\tau,...)-J(\theta_1,...,\theta_i+\tau,...)}{2\tau}\sim d\theta^{[i]}=\frac{\partial J}{\partial \theta_i}$

> Don’t use the approximation in gradient checking in training – only to debug
>
> If algorithm fails ***Gradient Checking***, just look at components to try to debug
>
> Remember your ***Regularization Term***
>
> Don’t use ***Gradient Checking*** together with ***Dropout Regulation***
>
> Run at random initialization, perhaps again after some training

