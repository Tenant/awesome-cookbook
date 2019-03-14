# Machine Learning

[Lecture video](http://open.163.com/special/opencourse/machinelearning.html)

[Lecture Note](https://onenote-shadow-dataset.oss-cn-beijing.aliyuncs.com/Course/CS229%20Lecture%20Notes.pdf)

[Syllabus](http://cs229.stanford.edu/syllabus.html)

## 1. Linear Regression

### 1.1 Least Mean Squares(LMS)

Tag: ==Supervised Learning==, ==Parametric Algorithm==.

==Predict function==: $h(x)=\sum_{i=0}^n\theta_ix_i=\theta^Tx$, where $n$ is the number of variable features.

==Cost function==: $J(\theta)=1/2\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$, where $m$ is the number of training examples.

Obviously, $J(\theta )$ is a ==convex quadratic function==.

#### Closed-form Solution

$$\begin{align}J(\theta)=\frac{1}{2}(X\theta-y)^T(X\theta-y)\\=\frac{1}{2}(tr\theta^TX^TX\theta-2try^TX\theta+y^Ty)\end{align}$$

$$\begin{align}\nabla_\theta J(\theta)=\frac{1}{2}\nabla_\theta (tr\theta^TX^TX\theta-2try^TX\theta+y^Ty)\\=\frac{1}{2}(2X^TX\theta-2X^Ty)\\=X^TX\theta-X^Ty\end{align}$$

Set $\nabla_\theta (\theta)=0$, then $\theta=(X^TX)^{-1}X^Ty$.

#### Why LMS?

Assume $y^{(i)}=\theta^TX^{(i)}+\epsilon^{(i)}$, where $\epsilon^{(i)}$ is non-linear error and $\epsilon^{(i)}\sim N(0,\sigma^2)$.

The maximum likelihood estimation is:

 $$\begin{align}L(\theta)=L(\theta;X,y)\\=P(y|X;\theta)\\=\prod_{i=1}^mp(y^{(i)}|x^{(i)};\theta)\\=\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}}\end{align}$$

$$\begin{equation}\ell(\theta )=log{L(\theta)}=m\times log (\frac{1}{\sqrt{2\pi}\sigma})-\frac{1}{2\sigma^2}\sum_{i=1}^m(y^{(i)}-\theta^Tx^{(i)})^2\end{equation}$$

Therefore, maximizing $\ell(\theta)$ equals to minimizing $\frac{1}{2}\sum_{i=1}^m(y^{(i)}-\theta^Tx^{(i)})^2$.

### 1.2 Locally Weighted Linear Regression(LWR)

Tag: ==Non-parametric Algorithm==

This algorithm does the following:

> Fit $\theta$ to minimize $\sum_{i=1}^mw^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2$, where $w^{(i)}=exp(-\frac{(x^{(i)}-x)^2}{2\tau^2 })$, and $\tau$ is ==the bandwidth parameter==.
>
> Output $\theta^Tx$.

## 2. Logistic Regression

Tag: ==Binary Classification==

==Predict Function==: $h_{\theta}(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$.

> $g(z)=\frac{1}{1+e^{-z}}$.
>
> $g'(z)=g(z)(1-g(z))$.

## 3. Digression: The Perceptron Learning Algorithm

## 4. Generalized Linear Models(GLMs)

### 4.1 The Exponential Family

We say that a class of distribution is  in the exponential family if it can be written in the form: $p(y;\eta)=b(y)exp(\eta^TT(y)-a(\eta))$, where $\eta$ is  ==the natural parameter==  of the distribution, $T(\eta)$ is the ==sufficient statistic==, and $a(\eta)$ is the log partion function. The quantity $e^{-a(\eta)}$ essentially plays the role of a normalization constant, that makes sure the distribution $p(y;\eta)$ sums/integrates over y to 1.

> For the distribution we consider, its sufficient statistic always satisfy that $T(y)=y$.

A fixed choice of $T$, $a$ and $b$ defines a family (or set) of distributions that is parameterized by $\eta$; as we vary $\eta$, we then get different distributions within this specific family.

#### 4.1.1 The Bernoulli Distribution

Specify a distribution over $y\in{0,1}$:

> $p(y=1;\phi)=\phi$
>
> $p(y=0;\phi)=1-\phi$

So that:

$$p(y;\phi)=\phi^y(1-\phi)^{1-y}=exp(y \log \frac{\phi}{1-\phi}+\log(1-\phi))$$.

To formulate the Bernoulli distribution as an exponential family distribution, we have:

> $\eta=\log\frac{\phi}{1-\phi}$
>
> $T(y)=y$
>
> $a(\eta)=-\log(1-\phi)=\log(1+e^\eta)$
>
> $b(y)=1$

#### 4.1.2 The Gaussian Distribution

Considering the value of $\sigma^2$ has no effect on the derivation of linear regression, we can choose an arbitrary value for $\sigma^2$ without changing anything. To simplify the derivation, set $\sigma^2=1$.

$p(y;\mu)=\frac{1}{\sqrt{2\pi}}exp(-\frac{1}{2}(y-\mu)^2)=\frac{1}{\sqrt{2\pi}}exp(-\frac{1}{2}y^2)\cdot exp(\mu y-\frac{1}{2}\mu^2)$

To formulate the Gaussian distribution as an exponential family distribution, we have:

> $\eta=\mu$
>
> $T(y)=y$
>
> $a(\eta)=\frac{\mu^2}{2}=\frac{\eta^2}{2}$
>
> $b(y)=\frac{1}{\sqrt{2\pi}}exp(\frac{y^2}{2})$

#### 4.1.3 The Poisson Distribution

#### 4.1.4 The Beta Distribution

#### 4.1.5 The Dirichletian Distribution

### 4.2 Construct GLMs

To derive a GLM, we make the following three assumptions about the conditional distribution of y given x: