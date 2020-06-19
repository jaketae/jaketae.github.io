---
title: Bayesian Linear Regression
mathjax: true
toc: true
categories:
  - study
tags:
  - bayesian
  - regression
  - linear_algebra
---

In today's post, we will take a look at Bayesian linear regression. Both [Bayes] and [linear regression] should be familiar names, as we have dealt with these two topics on this blog before. The Bayesian linear regression method is a type of linear regression approach that borrows heavily from Bayesian principles. The biggest difference between what we might call the vanilla linear regression method and the Bayesian approach is that the latter provides a probability distribution instead of a point estimate. In other words, it allows us to reflect uncertainty in our estimate, which is an additional dimension of information that can be useful in many situations. 

By now, hopefully you are fully convinced that Bayesian linear regression is worthy of our intellectual exploration. Let's take a deep dive into Bayesian linear regression, then see how it works out in code using the `pymc3` library.

# Bayesian Linear Regression

In this section, we will derive the formula for Bayesian linear regression step-by-step. If you are feeling rusty on linear algebra or Bayesian analysis, I recommend that you go take a quick review of these concepts before proceeding. Note that I borrowed heavily from this [video] for reference.

## Model Assumption

For any regression problem, we first need a data set. Let $D$ denote this pre-provided data set, containing $n$ entries where each entry contains an $m$-dimensional vector and a corresponding scalar. Concretely, 

$$D = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$$

where

$$x \in \mathbb{R}^m, \ y \in \mathbb{R}$$

The goal of Bayesian linear regression is to find the predictive posterior distribution for $\hat{y}$. This is where the difference between Bayesian linear regression and the normal equation method becomes most apparent. Whereas vanilla linear regression only gives us a single point estimate given an input vector, Bayesian linear regression gives an entire distribution. For the purposes of our demonstration, we will define the predictive posterior to take the following form as shown below, with precision $a$ pre-given. Precision is simply the reciprocal of variance and is commonly used as an alternative way of parametrizing Gaussian distributions. 

$$\hat{y} \sim \mathcal{N}(w^Tx, a^{-1}) \tag{1}$$

In other words, we assume the model

$$P(\hat{y}) = \text{exp}\left(- \frac{a}{2} (\hat{y} - w^Tx)^2 \right)$$

Our goal will be to derive a posterior for this distribution by performing Bayesian inference on $w$, which corresponds to the slope of the linear regression equation,

$$\hat{y} = w^Tx + \epsilon \tag{2}$$

where $\epsilon$ denotes noise and randomness in the data, thus affecting our final prediction.

## Prior Distribution

To begin Bayesian inference on parameter $w$, we need to specify a prior. Our uninformed prior will look as follows.

$$w \sim \mathcal{N}(0, b^{-1}I) \tag{3}$$ 

where $b$ denotes precision, the inverse of variance. Note that we have a diagonal covariance matrix in place of variance, the distribution for $w$ will be a [multivariate Gaussian]. 

The next ingredient we need for our recipe is the likelihood function. Recall that likelihood can intuitively be understood as an estimation of how likely it is to observe the given data points provided some parameter for the true distribution of these samples. The likelihood can easily be computed by referencing back to equation (1) above. Note that the dot product of $y - Aw$ with itself yields the sum of the exponents, which is precisely the quantity we need when computing the likelihood. 

$$P(D \vert w) \propto \text{exp}\left(-\frac{a}{2} (y - Aw)^T (y - Aw) \right) \tag{4}$$

where $A$ is a design matrix given by 

$$A = \begin{pmatrix} {x_1}^T \\ {x_2}^T \\ \vdots \\ {x_n}^T \end{pmatrix}$$

and $y$ is a column vector given by

$$y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$$

## Posterior Distribution

Before calculating the posterior, let's recall what the big picture of Bayesian inference looks like. 

$$P(\theta \vert x) = \frac{P(x \vert \theta) P(\theta)}{P(x)} \tag{5}$$

where $\theta$ denotes the parameter of interest for inference. In plain terms, the proposition above can be written as

$$\begin{align*} P_\text{Posterior} &= \frac{\mathcal{L} \cdot P_\text{Prior}}{P_\text{Evidence}} \\ &\propto \mathcal{L} \cdot P_\text{Prior} \end{align*} \tag{6}$$

In other words, the posterior distribution can be obtained by calculating the product of the prior distribution and the likelihood function. In many real-world cases, this process can be intractable, but because we are dealing with two Gaussian distributions, the property of [conjugacy] ensures that this problem is not only tractable, but also that the resulting posterior would also be Gaussian. 

$$\begin{align*} P(w \vert D) &\propto P(D \vert w) P(w) \\ &\propto \text{exp}\left(-\frac{a}{2} (y - Aw)^T (y - Aw) -\frac{b}{2} w^Tw \right) \\ &\propto \text{exp}\left(-\frac12 (a(y - Aw)^T (y - Aw) + bw^Tw) \right) \\ &\propto \text{exp}\left(-\frac12 (ay^Ty - 2aw^TA^Ty + w^T(aA^TA + bI)w) \right) \end{align*} \tag{7}$$

Although this may not be immediately apparent, observe that the exponent is a quadratic that follows the form

$$- \frac12 (w - \mu)\Lambda(w - \mu)^T$$

after making appropriate substitutions

$$\Lambda = aA^TA + bI, \ \mu = a\Lambda^{-1}A^Ty \tag{8}$$

Therefore, we know that the posterior for $w$ is indeed Gaussian, parameterized as follows:

$$w \sim \mathcal{N}(\mu, \Lambda^{-1}) \tag{9}$$

## Digression on MAP versus MLE

Let's try to obtain the MAP estimate of of $w$, *i.e.* simplify $\mu$

$$\begin{align} w_{MAP} &= \mu \\ &= a\Lambda^{-1}A^Ty \\ &= a(aA^TA + bI)^{-1}A^Ty \\ &= \left(A^TA + \frac{b}{a}I \right)^{-1}A^{T}y \end{align} \tag{10}$$

Notice the similarity with the MLE estimate, which is the solution to normal equation, which I otherwise referred to as vanilla linear regression:

$$w_{MLE} = (A^TA)^{-1}A^Ty \tag{11}$$

This is no coincidence: in a previous post on [MAP and MLE], we observed that the MAP and MLE become identical when we have a uniform prior. In other words, the only cause behind the divergence between MAP and MLE is the existence of a prior distribution. We can thus consider the additional term in (10) absent in (11) as a vestige of the prior we defined for $w$. MAP versus MLE is a recurring theme that appears throughout the paradigmatic shift from frequentist to Bayesian, so it merits discussion.

## Predictive Distribution

Now that we have a posterior distribution for $w$ which we can work with, it's time to derive the predictive distribution. We go about this by marginalizing $w$ using the property of conditional probability, as illustrated below.

$$\begin{align*} P(\hat{y} \vert x, D) &= \int P(\hat{y} \vert x, D, w) P(w \vert x, D) \, dw \\ &= \int \mathcal{N}(\hat{y} \vert w^Tx, a^{-1}) \mathcal{N}(w \vert \mu, \Lambda^{-1}) \, dw \\ &\propto \int \text{exp} \left(- \frac{a}{2}(\hat{y} - w^Tx)^2 - \frac12(w - \mu)^T \Lambda (w - \mu) \right) \, dw \\ &\propto \int \text{exp} \left(- \frac{a}{2}(\hat{y}^2 - 2w^Tx\hat{y} + (w^Tx)^2) - \frac12(w^T \Lambda w - 2w^T \Lambda \mu + \mu^T \Lambda \mu) \right) \, dw \\ &\propto \int \text{exp}\left(-\frac12 (a\hat{y}^2 - 2w^Tx\hat{y}a + aw^Txx^Tw + w^T \Lambda w - 2w^T \Lambda \mu) \right) \, dw \\ &\propto \int \text{exp}\left(-\frac12 (w^T(axx^T + \Lambda)w - 2w^T(x\hat{y}a + \Lambda \mu) + a\hat{y}^2) \right) \, dw \end{align*} \tag{12}$$

This may seem like a lot, but most of it was simple calculation and distributing vectors over parentheses. It's time to use the power of conjugacy again to extract a normal distribution out of the equation soup. Let's complete the square of the exponent according to the Gaussian form

$$- \frac12 (w - m)^T L (w - m)$$

after making the appropriate substitutions

$$L = axx^T + \Lambda, \ m = L^{-1}(x\hat{y}a + \Lambda \mu) \tag{13}$$

Again, observing this is not a straightforward process, especially if we had no idea what the final distribution is going to look like. However, given that the resulting predictive posterior will take a Gaussian form, we can backtrack using this knowledge to obtain the appropriate substitution parameters in (13). Continuing, 

$$\begin{align*} P(\hat{y} \vert x, D) &\propto \int \text{exp}\left(- \frac12 (w^T(axx^T + \Lambda)w - 2w^T(x\hat{y}a + \Lambda \mu) + a\hat{y}^2) \right) \, dw \\ &\propto \int \text{exp}\left (- \frac12 (w^TLw - 2w^TLm + m^TLm - m^TLm + a\hat{y}^2) \right) \, dw \\ &\propto \int \text{exp}\left(- \frac12 \left((w - m)^T L (w - m) - m^TLm + a\hat{y}^2 \right) \right) \, dw \\ &\propto \text{exp}\left(- \frac12 \left(a\hat{y}^2 - m^TLm \right) \right) \int \text{exp}\left(- \frac12 (w - m)^T L (w - m) \right) \, dw \end{align*} \tag{14}$$

where the last equality stands because we can pull out terms unrelated to $w$ by considering them as constants. Why do we bother to pull out the exponent? This is because the integral of a probability density function evaluates to 1, leaving us only with the exponential term outside the integral. 

$$P(\hat{y} \vert x, D) \propto \text{exp}\left(- \frac12 \left(a\hat{y}^2 - m^TLm \right) \right) \tag{15}$$

To proceed further from here, let's take some time to zoom in on $m^TLm$ for a second. Substituting $m$, we get

$$\begin{align*} m^TLm &= (a\hat{y}x + \Lambda \mu)^TL^{-1}LL^{-1}(a\hat{y}x + \Lambda \mu) \\ &= (a\hat{y}x + \Lambda \mu)^TL^{-1}(a\hat{y}x + \Lambda \mu) \\ &= (a^2x^TL^{-1}x)\hat{y}^2 + 2(ax^TL^{-1}\Lambda \mu)\hat{y} + \mu^T \Lambda L^{-1} \mu \end{align*} \tag{16}$$

We can now plug this term back into (15) as shown below.  

$$\begin{align*} P(\hat{y} \vert x, D) &\propto \text{exp}\left(- \frac12 \left(a\hat{y}^2 - m^TLm \right) \right) \\ &\propto \text{exp}\left(- \frac12 \left(a\hat{y}^2 - (a^2x^TL^{-1}x)\hat{y}^2 + 2(ax^TL^{-1}\Lambda \mu)\hat{y} + \mu^T \Lambda L^{-1} \mu \right) \right) \\ &\propto \text{exp}\left(- \frac12 \left((a - a^2x^TL^{-1}x)\hat{y}^2 - 2(ax^TL^{-1}\Lambda \mu)\hat{y} - \mu^T \Lambda L^{-1} \mu \right) \right) \end{align*} \tag{17}$$

Although it may seem as if we made zero progress by unpacking $m^TLm$, this process is in fact necessary to complete the square of the exponent according to the Gaussian form

$$- \frac12 k(\hat{y} - u)^2$$

after making the substitutions

$$k = a- a^2x^TL^{-1}x, \ u = \frac{1}{k}(ax^TL^{-1}\Lambda \mu) \tag{18}$$

By now, you should be comfortable with this operation of backtracking a quadratic and rearranging it to complete the square, as it is a standard operation we have used in multiple parts of this process. 

Finally, we have derived the predictive distribution in closed form:

$$\hat{y} \sim \mathcal{N}\left(u, \frac{1}{k} \right) \tag{19}$$

With more simplification using the , it can be shown that

$$u = \mu^Tx, \ \frac{1}{k} = \frac{1}{a} + x^T \Lambda^{-1} x \tag{20}$$

And there's the grand formula for Bayesian linear regression! This result tells us that, if we were to simply get the best point estimate of the predicted value $\hat{y}$, we would simply have to calculate $\mu^Tx$, which is the tranpose product of the MAP estimate of the weights and the input vector! In other words, the answer that Bayesian linear regression gives us is not so much different from vanilla linear regression, if we were to reduce the returned predictive probability distribution into a single point. But of course, doing so would defeat the purpose of performing Bayesian inference, so consider this merely an intriguing food for thought. 

# Implementation in PyMC3

As promised, we will attempt to visualize Bayesian linear regression using the `pymc3` library. Doing so will not only be instructive from a perspective of honing probabilistic programming skills, but also help us better understand and visualize Bayesian inference invovled in linear regression as explored in the context of this article. Note that, being a novice in `pymc3`, I borrowed heavily from this [resource] available on the `pymc3` official documentation. 

First, let's begin by importing all necessary modules. 


```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
from pymc3 import *
```

Let's randomly generate two hundred data points to serve as our toy data set for linear regression. 


```python
size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, 200)
true_reg_line = true_intercept + true_slope * x
y = true_reg_line + np.random.normal(scale=.5, size=size)

data = dict(x=x, y=y)
```

Below is a simple visualization of the generated data points alongside the true line which we will seek to approximate through regression. 


```python
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(xlabel='x', ylabel='y')
ax.scatter(x, y, label='Sampled Data', color='skyblue')
ax.plot(x, true_reg_line, label='True Regression Line', color='gold', lw='2')
plt.legend()
plt.show()
```


<img src="/assets/images/2020-01-20-bayesian-regression_files/2020-01-20-bayesian-regression_18_0.svg">


Now is the time to use the `pymc3` library. In reality, all of the complicated math we combed through reduces to an extremely simple, single-line command shown below. Under the hood, the `pymc3` using variations of random sampling to produce an approximate estimate for the predictive distribution. 


```python
with Model() as model: 
    glm.GLM.from_formula('y ~ x', data)
    trace = sample(3000, cores=2)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sd, x, Intercept]
    Sampling 2 chains: 100%|██████████| 7000/7000 [00:04<00:00, 1707.33draws/s]


Now that the trace plot is ready, let's see what the estimated vallues are like. We drop the first hundred sampled values may have been affected by a phenomena known as [burn-in]. Intuitively, the sampler needs some time to stabilize around the mean value, which is why the first few samples may contain more noise and provide information of lesser value compared to the rest. 


```python
plt.figure(figsize=(7, 7))
traceplot(trace[100:])
plt.tight_layout()
plt.show()
```

<img src="/assets/images/2020-01-20-bayesian-regression_files/2020-01-20-bayesian-regression_22_2.svg">


We see two lines for each plot because the sampler ran over two chains by default. What do those sampled values mean for us in the context of linear regression? Well, let's plot some sampled lines using the `plot_posterior_predictive_glm` function conveniently made available through the `pymc3` library. 


```python
plt.figure(figsize=(7, 7))
plt.scatter(x, y, label='Sampled Data', color='skyblue')
plot_posterior_predictive_glm(trace, samples=100,
                              label='Posterior Predictive Regression Lines', color='silver')
plt.plot(x, true_regression_line, label='True Regression line', lw=3, color='gold')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```


<img src="/assets/images/2020-01-20-bayesian-regression_files/2020-01-20-bayesian-regression_24_0.svg">


We see that the gray lines, sampled by `pymc3`, all seem to be a good estimate of the true regression line, colored in gold. We might also notice that the sampled regression lines seem to stay below the true regression line for smaller values of $x$. This is because we have more samples beneath the true regression line that we have above it. Bayesian linear regression is able to account for such variations in data and uncertainty, which is a huge advantage over the simple MLE linear regression method. 


# Conclusion

The true power of Bayesian linear regression might be summarized as follows: instead of returning just a single line using the MLE weight estimate of data, Bayesian linear regression models the entire data set to create a distribution of linear functions so to speak, allowing us to sample from that distribution to obtain sample linear regression lines. This is an approach that makes much more sense, since it allows us to take into account the uncertainty in our linear regression estimate. The reason why the normal equation method is unable to capture this uncertainty is that---as you might recall from the derivation of the formula for vanilla linear regression---the tools we used did not involve any probabilistic modeling. Recall that we used only linear algebra and matrix calculus to derive the model for vanilla linear regression. Bayesian linear regression is more complicated in that it involves computations with probability density functions, but the end result is of course more rewarding. 

That's it for today! I hope you enjoyed reading this post. Catch you up in the next one. 

[Bayes]: https://jaketae.github.io/study/bayes/
[linear regression]: https://jaketae.github.io/study/linear-regression/
[video]: https://www.youtube.com/watch?v=dtkGq9tdYcI&list=PLD0F06AA0D2E8FFBA&index=59
[conjugacy]: https://en.wikipedia.org/wiki/Conjugate_prior
[MAP and MLE]: https://jaketae.github.io/study/map-mle/
[resource]: https://docs.pymc.io/notebooks/GLM-linear.html
[multivariate Gaussian]: https://jaketae.github.io/study/gaussian-distribution/
