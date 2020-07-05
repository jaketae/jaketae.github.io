---
title: The Exponential Family
mathjax: true
toc: true
categories:
  - study
tags:
  - statistics
  - probability_distribution
---

Normal, binomial, exponential, gamma, beta, poisson... These are just some of the many probability distributions that show up on just about any statistics textbook. Until now, I knew that there existed some connections between these distributions, such as the fact that a binomial distribution simulates multiple Bernoulli trials, or that the continuous random variable equivalent of the geometric distribution is the exponential. However, reading about the concept of the exponential family of distributions has lent me new insight, and I wish to share that renewed understanding on probability distributions through this post. 

# Understanding the Exponential Family

In this section, we will take a look at what the exponential family of distributions is all about. We will begin by laying out a few mathematical definitions, then proceed to see examples of probability distributions that belong to the exponential family. 

## Definition

To cut to the chase, the exponential family simply denotes a  group of probability distributions that satisfy a certain condition, namely that they can be factorized and parametrized into a  specific form, as show below:

$$
p_\theta(x) = h(x) \ \text{exp}\left(\eta(\theta)^T s(x) - k(\theta)\right) \tag{1}
$$

Here, $k(\theta)$ is a log noramlizing constant that ensures that the probability distribution integrates to 1.

$$
k(\theta) = \log\left(\int h(x) \ \text{exp}\left(\eta(\theta)^T s(x)\right) \, dx\right) \tag{2}
$$

There are other alternative forms that express the same factorization. One such variant that I prefer and find more intuitive uses a simple fractional approach for normalization instead of adding complications to the exponential term. For notational convenience, I will follow the fractional normalization approach shown below throughout this post. 

$$
p_\theta (x) = \frac{1}{z(\theta)} h(x) \ \text{exp}\left(\eta(\theta)^T s(x)\right) \tag{3}
$$

Before we proceed any further, it is probably a good idea to clarify the setup of the equations above. First, $x$ denotes a $d$-dimensional random variable of interest; $\theta$, a $k$-dimensional parameter that defines the probability distribution. $s(x)$ is known as the [sufficient statistic function](https://en.wikipedia.org/wiki/Sufficient_statistic). Below is a brief summary concerning the mappings of these different functions. 

$$
x \in \mathbb{R}^d, \ \theta \in \mathbb{R}^k, \ \eta_i: \Theta \to \mathbb{R}, \ s_i: \mathbb{R}^d \to \mathbb{R}, \ h: \mathbb{R}^d \to [0, \infty), \ z: \Theta \to (0, \infty)
$$

You will notice that I used $s_i$ and $\eta_i$ instead of $s$ and $\eta$ as shown in equation (3). This is because (3) assumes vectorization of these functions as follows.

$$
\eta(\theta) = \begin{pmatrix} \eta_1(\theta) \\ \eta_2(\theta) \\ \vdots \\ \eta_m(\theta) \end{pmatrix}, \ s(x) = \begin{pmatrix} s_1(x) \\ s_2(x) \\ \vdots \\ s_m(x) \end{pmatrix} \tag{4}
$$

We could have expressed (3) without vectorization, but doing so would be rather verbose.

$$
p_\theta (x) = \frac{1}{z(\theta)} h(x) \ \text{exp}\left(- \sum_{i = 1}^m \eta_i(\theta) s_i(x) \right) \tag{5}
$$

So we instead adhere to the vectorized convention in (3) throughout this post. 

## Examples of Exponential Family Distributions

As I hinted earlier, the exponential family covers a wide range of probability distributions, most PDFs and PMFs. In fact, most probability distributions that force themselves onto the page of statistics textbooks belong to this powerful family. Below is a non-comprehensive list of distributions that belong to the exponential family. 

Probability Density Functions

* Exponential
* Gaussian
* Beta
* Gamma
* Chi-squared

Probability Mass Functions

* Bernoulli
* Binomial
* Poisson
* Geometric
* Multinomial

Of course, there are examples of common distributions that do not fall under this category, such as the uniform distribution or the student $t$-distribution. This point notwithstanding, the sheer coverage of the exponential family makes it worthy of exploration and analysis. Also, notion of an exponential family itself is significant in that it allows us to frame problems in meaningful ways, such as through the notion of conjugate priors: if you haven't noticed, the distributions outlined above all have conjugate priors that also belong to the exponential family. In this sense, the exponential family is particularly of paramount importance in the field of Bayesian inference, as we have seen many times in previous posts. 

### Factorizing the Exponential Distribution

Let's concretize our understanding of the exponential family by applying factorization to actual probability distributions. The easiest example, as you might have guessed, is the exponential distribution. Recall that the formula for the exponential distribution is

$$
p_\theta(x) = \theta e^{- \theta x} I(x) \tag{6}
$$

where the indicator function, denoted as $I(x)$, takes the following form:

$$
I(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases} \tag{7}
$$

The indicator function is a simple  modification applied to ensure that the function is well-defined across the entire real number domain. Normally, we omit the indicator function since it is self-apparent, but for the sake of robustness in our analysis, I have added it here.

How can we coerce equation (6) to look more like (3), the archetypal form that defines the exponential family? Well, now it's just a matter of drag and match: by paying close attention to the variables, parameters, and the output of each function, we can reconstruct (3) to take the form of (6). The easeist starting point is to observe the exponent to identify $\eta$ and $s$, after which the rest of the surrounding functions can be inferred. The end result is presented below:

$$
\eta(\theta) = \theta, \ s(x) = -x, \ h(x) = I(x), \ z(\theta) = \frac{1}{\theta} \tag{8}
$$

After substituting each function with their prescribed value in (8), it isn't difficult to see that the exponential distribution can indeed by factorized according to the form outlined in (3). Although this is by no means a rigorous proof, we see not only the evident fact that the exponential distribution indeed belongs to the exponential family, but also that the factorization formula in (3) isn't just a complete soup of equations and variables. 

### Factorizing the Bernoulli Distribution

We can do the same for the Bernoulli distribution, which also falls under the exponential family. The formula for the Bernoulli distribution goes as follows:

$$
p_\theta(x) = \theta^x(1 - \theta)^{1 - x}I(x) \tag{9}
$$

Again, I have added a very simple indicator function to ensure that the the probability mass function is well-defined across the entire real number line. Again, the indicator function is a simple boolean gate function that checks whether $x$ is an element within a set of zero and one:

$$
I(x) = \begin{cases} 1 & x \in \{0, 1\} \\ 0 & x \not \in \{0, 1\} \end{cases} \tag{10}
$$

Factorizing the Bernoulli is slightly more difficult than doing the same for the exponential distribution, largely because it is not apparent from (9) how factorization can be achieved. For example, we do not see any exponential term embedded in (9) as we did in the case of the exponential distributions. Therefore, a simple one-to-one correspondence cannot be identified. The trick to get around this problem is to introduce a log transformation, then reapplying an exponential. In other words, 

$$
e^{\log(p_\theta(x))} = p_\theta(x) = I(x) \ \text{exp}\left(x\log(\theta) + (1 - x)\log(1 - \theta)\right) \tag{11}
$$

By applying this manipulation, we can artificially create an exponential term to more easily coerce (9) into the factorization mold. Specifically, observe that the power of the exponent can be expressed as a dot product between two vectors, each parameterized by $x$ and $\theta$ , respectively. This was the hard part: now, all that is left is to configure the rest of the functions to complete the factorization. One possible answer is presented below:

$$
\eta(\theta) = \begin{pmatrix} \log(\theta) & \log(1 - \theta) \end{pmatrix}^T, \\ s(x) = \begin{pmatrix} x & 1 - x \end{pmatrix}^T, \\ h(x) = I(x), \ z(\theta) = 1 \tag{12}
$$

By now, it should be sufficienty clear that the definition of the exponential family is robust enough to encompass at least the two probability distributions: the exponential and the Bernoulli. Although we do not go over other examples in this article, the exponential family is a well-defined set of probability distributions that, at thei core, are defined by a common structure. And as we will see in the next section, this underlying similarity makes certain calculations surprisingly convenient. 

# Maximum Likelihood Estimation in Canonical Form

In a previous post, we explorerd the notion of [maximum likelihood estimation](https://jaketae.github.io/study/likelihood/), and contrasted it with [maximum a posteriori estimation](https://jaketae.github.io/study/map-mle/). The fundamental question that maximum likelihood estimation seems to answer is: given some data, what parameter of a distribution best explains that observation? This is an interesting question that merits exploration in and of itself, but the discussion becomes a lot more interesting and pertinent in the context of the exponential family. 

## Canonical Form

Before diving into MLE, let's define what is known as the canonical form of the exponential family. Despite its grandiose nomenclature, the canonical form simply refers to a specific flavor of factorization scheme where

$$
\eta(\theta) = \theta \tag{13}
$$

in which case (3) simplifies to

$$
p_\theta(x) = \frac{1}{z(\theta)}h(x) \ \text{exp}\left(\theta^T s(x)\right) \tag{14}
$$

We will assume some arbitrary distribution in the exponential family following this canonical form to perform maxmimum likelihood estimation. 

## Maximum Likelihood Estimation

Much like in the previous post on maximum likelihood estimation, we begin with some data set of $n$ independent and identically distributed observations. This is going to be the setup of the MLE problem. 

$$
D = (x_1, x_2, \dots, x_n), \ x \in \mathbb{R}^d
$$

Given this dataset, the objective of maximum likelihood estimation is to identify some parameter $\theta$ that maximizes the likelihood, *i.e.* the probability of observing these data points under a probability distribution defined by $\theta$. In other words, 

$$
\theta_{MLE} = \mathop{\rm arg\,max}\limits_{\theta \in \Theta} p(D \vert \theta) \tag{15}
$$

How do we identify this parameter? Well, the go-to equipment in a mathematician's arsenal for an optimization problem like this one is calculus. Recall that our goal is to maximize the likelihood function, which can be calculated as follows:

$$
\begin{align} p(D \vert \theta) &= \prod_{i = 1}^n p(x_i \vert \theta) \\ &= \frac{1}{z(\theta)^n}\prod_{i = 1}^n h(x_i) \ \text{exp}\left(\theta^T \sum_{i = 1}^n s(x_i)  \right) \\ &= \frac{1}{z(\theta)^n}\prod_{i = 1}^n h(x_i) \ \text{exp}\left(\theta^T s(D) \right) \end{align} \tag{16}
$$

The first equality stands due to the assumption that all data are independent and identically distributed. 

Maximizing (16) is a complicated task, especially because we are dealing with a large product. Products aren't bad, but we typically prefer sums because they are easier to work with. A simple hack that we almost always use when dealing with maximum likelihood, therefore, is to apply a log transformation to calculate the log likelihood, since the logarithm is a monotonically increasing function. In other words, 

$$
\theta_{MLE} = \mathop{\rm arg\,max}\limits_{\theta \in \Theta} \log(p(D \vert \theta)) \tag{17}
$$

What does the log likelihood look like? Well, all we have to do is to apply a log function to (16), which yields the following result.

$$
\log(p(D \vert \theta)) = -n\log(z(\theta)) + \theta^T s(D) + \sum_{i = 1}^n \log(h(x_i)) \tag{18}
$$

Maximizing the log liklihood can be achieved by setting the gradient to zero, as the gods of calculus would tell us. As you might recall from a previous post on some very basic [matrix calculus](https://jaketae.github.io/study/linear-regression/), the gradient is simply a way of packaging derivatives in a multivariate context, typically involving vectors. If any of this sounds unfamilar, I highly recommend that you check out the linked post. 

We can compute the partial derivative of the log likelihood function with respect to $\theta_i$ as shown below. Observe that the last term in (18) is eliminated because it is a constant with respect to $\theta_i$.

$$
\frac{\partial \log(p(D \vert \theta))}{\partial \theta_j} = - n \frac{\partial \log(z(\theta))}{\partial \theta_j} + s_j(D) \tag{19}
$$

This is a good starting point, but we still have no idea how to derive the log of $z(\theta)$. To go about this problem, we have to derive an expression for $z$. Recall from the definition of the exponential family that $z$ is a normalizing constant that exists to ensure that the probability function integrates to one. In other words, 

$$
\int_{X \in \mathbb{R}^d} p_\theta(x) = \int \frac{h(x) \ \text{exp}\left(\theta^T s(x)\right)}{z(\theta)} \, dx = 1 \tag{20}
$$

This necessarily implies that 

$$
z(\theta) = \int h(x) \ \text{exp}\left(\theta^T s(x)\right) \, dx \tag{21}
$$

Now that we have an expression for $z(\theta)$ to work with, let's try to compute the derivative term we left unsolved in (19). 

$$
\begin{align} \frac{\partial \log(z(\theta))}{\partial \theta_j} &= \frac{1}{z(\theta)} \int \frac{\partial \left(h(x) \ \text{exp}\left(\theta^T s(x)\right) \right)}{\partial \theta_j} \, dx \\ &= \frac{1}{z(\theta)} \int h(x) s_j(x) \ \text{exp}\left(\theta^T s(x) \right) \, dx \\ &= \int  s_j(x) \frac{h(x) \ \text{exp}\left(\theta^T s(x) \right)}{z(\theta)} \, dx \\ &= \int s_j(x) p(x \vert \theta) \, dx \\ &= \mathbb{E}_\theta \left[s_j(X)\right] \end{align} \tag{22}
$$

The first and second equalities stand due to the chain rule, and the third equality is a simple algebraic manipulation that recreates the probability function within the integral, allowing us to ultimately express the partial derivative as an expected value of $s_j$ for the random variable $X$. This is a surprising result, and a convenient one indeed, because we can now use this observation to conclude that the gradient of the log likelihood function is simply the expected value of the sufficient statistic. 

$$
\nabla_\theta \log(z(\theta)) = \mathbb{E}_\theta\left[s(X)\right] \tag{23}
$$

Therefore, starting again from (19), we can continue our calculation of the gradient and set the quantity equal to zero to calculate the MLE estimate of the parameter. 

$$
\begin{align} \nabla_\theta \log(p(D \vert \theta)) &= - n \nabla_\theta \log(z(\theta)) + s(D) \\ &= - n \ \mathbb{E}_\theta \left[s(X)\right] + s(D) \\ &= 0 \end{align} \tag{24}
$$

It then follows that 

$$
\mathbb{E}_\theta \left[s(X)\right] = \frac{s(D)}{n} = \frac{1}{n}\sum_{i = 1}^n s(x_i) \tag{25}
$$

## Interpretation Under Exponential Distribution

How do we interpret the final result in equation (25)? It looks nice, simple, and concise, but what does it mean to say that the expected value of the sufficient statistic is the average of the sufficient statistic for each observed individual data points? To remove  abstractness, let's employ a simple example, the exponential distribution, and attempt to derive a clearer understanding of the final picture. 

Recall that the probability density function of the exponential distribution takes the following form according to the factorizations outlined below:

$$
p_\theta(x) = \theta e^{- \theta x} I(x) \tag{6} 
$$

$$
I(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases} \tag{7}
$$

$$
\eta(\theta) = \theta, \ s(x) = -x, \ h(x) = I(x), \ z(\theta) = \frac{1}{\theta} \tag{8}
$$

Computing the derivative of the log of the normalizing term $z(\theta)$ as we did in (22), 

$$
\begin{align} \nabla_\theta \log(z(\theta)) &= \frac{d}{d\theta}\left[\log\left(\frac{1}{\theta}\right) \right] \\ &= \frac{d}{d\theta}\left[- \log(\theta) \right] \\ &= - \frac{1}{\theta} \end{align} \tag{26}
$$

Because we know that the resulting quantity is the expected value of the sufficient statistic, we know that

$$
\mathbb{E}_\theta \left[s(X)\right] = \mathbb{E}_\theta [-X] = - \frac{1}{\theta} \implies \mathbb{E}_\theta [X] = \frac{1}{\theta} \tag{27}
$$

And indeed, this is true: the expected value of the random variable characterized by an exponential distribution is simply the inverse of the parameter defining that distribution. Note that the parameter for the exponential distribution is most often denoted as $\lambda$, in which case the expected value of the distribution would simply be written as $\mathbb{E}[X] = \frac{1}{\lambda}$. 

This is all great, but there is still an unanswered question lingering in the air: what is the MLE estimate of the parameter $\theta$ ? This moment is precisely when equation (25) comes in handy. Recall that

$$
\mathbb{E}_\theta \left[s(X)\right] = \frac{s(D)}{n} = \frac{1}{n}\sum_{i = 1}^n s(x_i) \tag{25}
$$

Therefore, 

$$
\mathbb{E}_\theta \left[s(X)\right] = - \frac{1}{\theta} = \frac{1}{n}\sum_{i = 1}^n s(x_i) = \frac{1}{n}\sum_{i = 1}^n - x_i \tag{28}
$$

Finally, we have arrived at our destination:

$$
\theta_{MLE} = \frac{1}{\frac{1}{n} \sum_{i = 1}^n x_i} \tag{29}
$$

We finally know how to calculate the parameter under which the likelihood of observing given data is maximized. The beauty of this approach is that it applies to all probability distributions that belong to the exponential family because our analysis does not depend on which distribution is in question; we started from the canonical form of the exponential family to derive a set of generic equations. This is the convenience of dealing with the exponential family: because they are all defined by the same underlying structure, the MLE equations hold general applicability.

# Conclusion

In this post, we explored the exponential family of distributions, which I flippantly ascribed the title "The Medici of Probability Distributions." This is obviously my poor attempt at an intellectual joke, to which many of you might cringe, but I personally think it somewhat captures the idea that many probability distributions that we see on the textbook are, in fact, surprisingly more related than we might think. At least to me, it wasn't obvious from the beginning that the exponential and the Bernoulli distributions shared the same structure, not to mention the wealth of other distributions that belong to the exponential family.  Also, the convenient factorization is what allowed us to perform an MLE estimation, which is an important concept in statistics with wide ranging applications. This post in no way claims to give a full, detailed view of the exponential family, but hopefully it gave you some understanding of what it is and why it is useful.

In the next post, we will take a look at maximum a posteriori estimation and how it relates to the concept of convex combinations. Stay tuned for more.