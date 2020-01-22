---
title: Convex Combinations and MAP
mathjax: true
toc: true
categories:
  - study
tags:
  - statistics
---

In a [previous post](https://jaketae.github.io/study/map-mle/), we briefly explored the notion of maximum a posteriori and how it relates to maximum likelihood estimation. Specifically, we derived a generic formula for MAP and explored how it compares to that for MLE. Today's post is going to be an interesting sequel to that story: by performing MAP on the univariate Gaussian, we will show how MAP can be interpreted as a [convex combination](https://en.wikipedia.org/wiki/Convex_combination), thus motivating a more intuitive understanding of what MAP actually entails under the hood. Let's jump right into it.

# MAP of Univariate Gaussian

The univariate Gaussian is a good example to work with because it is simple and intuitive yet also complex enough for meaningful analysis. After all, it is one of the most widely used probability distributions and also one that models many natural phenomena. With that justification firmly in mind, let's take a look at the setup of the MAP of the  mean for the univariate Gaussian. 

## Problem Setup

As always, we begin with some dataset of $n$ independent observations. In this case, because we are dealing with the univariate Gaussian, each observations will simply be a scalar instead of a vector. In other words, 


$$
D = (x_1, x_2, \dots, x_n), \ x \in \mathbb{R}
$$


Let's assume that the random variable is normally distributed according to some parameter $\theta$. We will assume that the standard deviation of the random variable is given as $\sigma$. We could have considered standard deviation to be a parameter, but since the goal of this demonstration is to conduct MAP estimation on the mean of the univariate Gaussian, we assume that the standard deviation is known. 


$$
X \sim \mathcal{N}(\theta, \sigma^2)
$$


Next, we have to define a prior distribution for the parameter. Let's say that $\theta$ is also normally distributed around some mean $\mu$ with a standard deviation of 1, as shown below.


$$
\theta \sim \mathcal{N}(\mu, 1)
$$


## Maximum A Posteriori

Recall that the goal of MAP is, as the same suggests, to maximize the posterior distribution. To derive the posterior, we need two ingredients: a prior and a likelihood function. We already have the first ingredient, the prior, as we have just defined it above. The last piece of the puzzle, then, is the likelihood function. Since we have assumed our data to be independently distributed, we can easily calculate the likelihood as follows:


$$
p(D \vert \theta) = p(x_1, x_2, \dots, x_n \vert \theta) = \prod_{i = 1}^n p(x_i \vert \theta) \tag{1}
$$


All that is left is to compute the posterior according to [Baye's formula](https://en.wikipedia.org/wiki/Bayes%27_theorem) for [Bayesian inference](https://jaketae.github.io/study/bayes/). We can thus calculate the MAP estimate of $\theta$ as shown below.


$$
\begin{align}\theta_{MAP} &= \mathop{\rm arg\,max}\limits_{\theta} p(\theta \vert D)\\ &= \mathop{\rm arg\,max}\limits_{\theta} \frac{p(D \vert \theta) p(\theta)}{p(D)} \\ &= \mathop{\rm arg\,max}\limits_{\theta} p(D \vert \theta) p(\theta) \\ &= \mathop{\rm arg\,max}\limits_{\theta} \log(p(D \vert \theta) p(\theta)) \\ &= \mathop{\rm arg\,max}\limits_{\theta} \log(p(D \vert \theta)) + \log(p(\theta)) \end{align} \tag{2}
$$


The second equality is due to proportionality, whereby $p(D)$ is independent of $\theta$ and thus can be removed from the argmax operation. The fourth equality is due to the monotonically increasing nature of the logarithmic function. We always love using logarithms to convert products to sums, because sums are almost always easier to work with than products, especially when it comes to integration or differentiation. If any of these points sounds confusing or unfamiliar, I highly recommend that you check out my articles on [MAP](https://jaketae.github.io/study/map-mle/) and [MLE](https://jaketae.github.io/study/likelihood/). 

To proceed, we have to derive concrete mathematical expressions for the log likelihood and the log prior. Recall the formula for the univariate Gaussian that describes our data:


$$
p(x \vert \theta) = \frac{1}{\sigma \sqrt{2 \pi}} \text{exp}\left(-{\frac{(x - \theta)^2}{2\sigma^2}}\right) \tag{3}
$$


Then, from (1), we know that the likelihood function is simply going to be a product of the univariate Gaussian distribution. More specifically, the log likelihood is going to be the sum of the logs of the Gaussian probability distribution function.


$$
\begin{align} \log(p(D \vert \theta)) &= \log(\prod_{i = 1}^n p(x_i \vert \theta)) \\ &= \sum_{i = 1}^n \log(p(x_i \vert \theta)) \\ &= n\log(\frac{1}{\sigma \sqrt{2 \pi}}) - \frac{1}{2\sigma^2}\sum_{i = 1}^n (x_i - \theta)^2 \end{align} \tag{4}
$$


There is the log likelihood function! All we need now is the log prior. Recall that the prior is a normal distribution centered around mean $\mu$ with standard deviation of 1. In PDF terms, this translates to 


$$
p(\theta) = \frac{1}{\sqrt{2 \pi}} \text{exp}\left(- \frac{1}{2}(\theta - \mu)^2 \right) \tag{5}
$$


The log prior can simply be derived by casting the logarithmic function to the probability distribution function.


$$
\log(p(\theta)) = \log \left(\frac{1}{\sqrt{2 \pi}}\right) - \frac{1}{2}(\theta - \mu)^2 \tag{6}
$$


Now we are ready to enter the maximization step of the sequence. To calculate the maximum of the posterior distribution, we need to derive the posterior and set the gradient equal to zero. For a more robust analysis, it would be required to show that the second derivative is smaller than 0, which is indeed true in this case. However, for the sake of simplicity of demonstration, we skip that process and move directly to calculating the gradient.


$$
\begin{align} \frac{\partial}{\partial \theta}\left[ \log(p(D \vert \theta)) + \log(p(\theta)) \right] &= \frac{\partial}{\partial \theta}\left[ \log(p(D \vert \theta)) \right] + \frac{d}{d \theta}\left[\log(p(\theta)) \right] \\ &= \frac{1}{\sigma^2} \left(\sum_{i = 1}^n x_i - n \theta \right) + (\mu - \theta) \\ &= \left(\sum_{i = 1}^n \frac{x_i}{\sigma^2} + \mu \right) - \left(\frac{n}{\sigma^2} + 1 \right) \theta \\ &= 0 \end{align} \tag{7}
$$


Let's rearrange the final equality in (7).


$$
\left(\frac{n}{\sigma^2} + 1 \right) \theta = \sum_{i = 1}^n \frac{x_i}{\sigma^2} + \mu \tag{8}
$$


From (8), we can finally derive an expression for $\theta_{MAP}$. This value of the parameter is one that which maximizes the posterior distribution. 


$$
\begin{align} \theta_{MAP} &= \frac{\sum_{i = 1}^n \frac{x_i}{\sigma^2} + \mu}{\left(\frac{n}{\sigma^2} + 1 \right)} \\ &= \frac{\sum_{i = 1}^n x_i + \sigma^2 \mu}{n + \sigma^2} \\ &= \frac{n}{n + \sigma^2}\bar{x} + \frac{\sigma^2}{n + \sigma^2}\mu \end{align} \tag{9}
$$


And we have derived the MAP estimate for the mean of the univariate Gaussian!



# Convex Combinations

Maximum a posteriori analysis is great and all, but what does the final result exactly tell us? While there might be many ways to interpret understand the result as derived in (9), one particular useful intuition to have relates to the concept of [convex combinations](https://en.wikipedia.org/wiki/Convex_combination). 

Simply put, a convex combination is a linear combination of different points or quantities in which the coefficients of the linear combinations add up to one. More concretely,

$$
\alpha_1 x_1 + \alpha_2 x_2 + \cdots + \alpha_n x_n, \ \sum_{i = 1}^n \alpha_i = 1 \tag{10}
$$

We can also imagine that $\alpha$ and $x$ are each $n$-dimensional vectors, and that a convex combination is simply a dot product of these two vectors given that the elements of $\alpha$ sum up to one.

Why did I suddenly bring up convex combinations out of no where? Well, it turns out that the result in (9) in fact an instance of a convex combination of two points satisfying the form 


$$
\alpha x + (1 - \alpha)y \tag{11}
$$


Indeed, it is not difficult to see that the coefficient of $\bar{x}$ and $\mu$ add up to 1, which is precisely the condition for a linear combination to be considered convex. Now here is the important part: the implication of this observation is that we can consider the MAP estimate of parameter $\theta$ as an interpolation, or more simply, some weighted average between $\bar{x}$ and $\mu$. This interpretation also aligns with the whole notion of Bayesian inference: our knowledge of the parameter is partially defined by the prior, but updated as more data is introduced. And as we obtain larger quantities of data, the relative importance of the prior distribution starts to diminish. Imagine that we have an infinite number of data points. Then, $\theta_{MAP}$ will soley be determined by the likelihood function, as the weight ascribed to the prior will decrease to zero. In other words, 


$$
\lim_{n \to \infty} \frac{n}{n + \sigma^2} = 1 \\ \lim_{n \to \infty} \frac{\sigma^2}{n + \sigma^2} = 0 \tag{12}
$$


Conversely, we can imagine how having no data points at all would cause the weight values to shift in favor of the prior such that no importance is ascribed to the MLE estimate of the parameter.

# Conclusion

In this short article, we reviewed the concept of maximum a posteriori and developed a useful intuition about its result from the perspective of convex combinations. Maximum a posteriori, alongside its close relative maximum likelihood estimation,  is an interesting topic that deserives our attention. Hopefully through this post, you gained a better understanding of what the result of an MAP estimation actually means from a Bayesian point of view: a weighted average between the prior mean and the MLE estimate, where the weight is determined by the number of data points at our disposal. Also, I just thought that the name "convex conbinations" is pretty cool. The fancier the name, the more desire to learn---it is a natural human instinct indeed. 

I hope you enjoyed reading this article. In the next post, we will continue our discussion by exploring the concept of conjugate priors, which is something that we have always assumed as true and glossed over, yet is arguably what remains at the very core of Bayesian analysis. 

