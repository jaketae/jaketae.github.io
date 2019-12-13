---
title: Dissecting the Gaussian Distribution
Mathjax: true
date: 2019-12-12
categories:
  - study
tags:
  - math
---

If there is one thing that the field of statistics wouldn’t be complete without, it’s probably normal distributions, otherwise referred to as “the bell curve.” The normal distribution was discovered and studied extensively by Carl Friedrich Gauss, which is why it is sometimes referred to as the Gaussian distribution. 

We have seen Gaussian distributions before in this blog, specifically on [this post] on likelihood and probability. However, normal distribution was introduced merely as an example back then. Today, we will put the Gaussian distribution on stage under undivided spotlight. Of course, it is impossible to cover everything about this topic, but it is my goal to use the mathematics we know to derive and understand this distribution in greater detail. Also, it’s just helpful to brush up on some multivariable calculus in a while.

# Deriving the Univariate Gaussian

Let’s start with the simplest case, the univariate Gaussian distribution. The “univariate” part is just a fancier way of saying that we will dealing be dealing with one-dimensional random variables, *i.e.* the distribution is going to be plotted on a two-dimensional $$xy$$ plane. We make this seemingly trivial distinction to distinguish it from the multivariate Gaussian, which can be plotted on three-dimensional space or beyond. We’ll take a look at the multivariate normal distribution in a later section. For now, let’s derive the univariate case. 

## The Definition

One of the defining properties of data that are said to be normally distributed when the rate at which the frequencies decrement is proportional to its distance from the mean and the frequencies themselves. Concretely, this statement might be translated as

$$\frac{df}{dx} = -k(x - \mu)f(x)$$

We can separate the variables to achieve the following expression:

$$\frac{df}{f} = -k(x - \mu) dx$$

Integrating both sides yields

$$\int \frac{1}{f} \, df = \int -k(x - \mu) \, dx$$

$$\ln(f) = \frac{-k(x - \mu)^2}{2} + C$$

Let’s get rid of the logarithm by exponentiating both sides. 

$$f = e^{\frac{-k(x - \mu)^2}{2} + C}$$

That’s an ugly exponent. But we can make things look better by observing that the constant term $$C$$ can be brought down as a coefficient, since

$$e^{\frac{-k(x - \mu)^2}{2} + C} = e^{- \frac{k}{2}(x - \mu)^2} \cdot e^C = A e^{- \frac{k}{2}(x - \mu)^2}$$

where we make the substitution $$A = e^C$$. Now, the task is to figure out what the constants $$A$$ and $$k$$ are. There is one constraint equation that we have not used yet: the integral of a probability distribution function must converge to 1. In other words, 

$$A \int_{- \infty}^{\infty} e^{- \frac{k}{2}(x - \mu)^2} \, dx = 1 \tag{1}$$

Now we run into a problem. Obviously we cannot calculate this integral as it is. Instead, we need to make a clever substitution. Here’s a suggestion: how about we get rid of the complicated exponential through the substitution

$$u = \sqrt{\frac{k}{2}} (x - \mu)$$

Then, it follows that 

$$du = \sqrt{\frac{k}{2}} dx$$

$$dx = \sqrt{\frac{2}{k}} du$$

Therefore, the integral in (1) now collapses into

$$A \sqrt{\frac{2}{k}} \int_{- \infty}^{\infty} e^{-u^2} \, du = 1 \tag{2}$$

Now that looks marginally better. But we have a very dirty constant coefficient at the front. Our natural instinct when we see such a square root expression is to square it. What’s nice about squaring in this case is that the value of the expression is going to stay unchanged at 1. 

$$ \frac{2A^2}{k} (\int_{- \infty}^{\infty} e^{-u^2} \, du) (\int_{- \infty}^{\infty} e^{-u^2} \, du) = 1$$

Because the two integrals are independent, *i.e.* calculating one does not impact the other, we can use two different variables for each integral. For notational convenience, let’s use $$x$$ and $$y$$.

$$ \frac{2A^2}{k} (\int_{- \infty}^{\infty} e^{-x^2} \, dx) (\int_{- \infty}^{\infty} e^{-y^2} \, dy) = 1$$

We can combine the two integrals to form an iterated integral of the following form:

$$\frac{2A^2}{k} \int_{- \infty}^{\infty} \int_{- \infty}^{\infty} e^{-(x^2 + y^2)} \, dx \, dy = 1$$

The term $$(x + y)^2$$ rings a bell, and that bell sounds like circles and therefore polar coordinates. Let’s implement a quick change of variables to move to polar coordinates.

$$\frac{2A^2}{k} \int_{0}^{2 \pi} \int_0^\infty r e^{-r^2} \, dr \, d\theta = 1 \tag{3}$$

Now we have something that we can finally integrate. Using the chain rule in reverse, we get

$$\frac{2A^2}{k} \int_{0}^{2 \pi} \left[-\frac12 e^{-r^2}\right]_0^\infty \, d\theta = \frac{A^2}{k} \int_0^{2\pi} \, d\theta = 1$$

We can consider there to be 1 in the integrand and continue our calculation. The result:

$$ \frac{2 \pi A^2}{k} = 1 \tag{4}$$

From (4), we can express $$A$$ in terms of $$k$$:

$$A = \sqrt{\frac{k}{2 \pi}}$$

After applying the substitution, now our probability density function looks as follows:

$$f = \sqrt{\frac{k}{2 \pi}} e^{- \frac{k}{2}(x - \mu)^2} \tag{5}$$

To figure out what $$k$$ is, let’s try to find the variance of $$x$$, since we already know that the variance should be equal to $$\sigma$$. In other words, from the definition of variance, we know that 

$$\sigma^2 = \int_{- \infty}^\infty (x - \mu)^2 f(x) \, dx$$

Using (5), we get

$$\sigma^2 = \sqrt{\frac{k}{2 \pi}} \int_{- \infty}^\infty (x - \mu)^2 e^{- \frac{k}{2}(x - \mu)^2} \, dx$$

We can use integration by parts to evaluate this integral.

