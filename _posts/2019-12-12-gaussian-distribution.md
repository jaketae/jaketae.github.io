---
title: Dissecting the Gaussian Distribution
mathjax: true
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

$$\sqrt{\frac{k}{2 \pi}} \int_{- \infty}^\infty (x - \mu)^2 e^{- \frac{k}{2}(x - \mu)^2} \, dx = \sqrt{\frac{k}{2 \pi}}\left[-\frac{1}{k} (x - \mu) e^{- \frac{k}{2}(x - \mu)^2} \right]_{-\infty}^\infty + \frac{1}{k} \sqrt{\frac{k}{2 \pi}} \int_{- \infty}^\infty e^{- \frac{k}{2}(x - \mu)^2} \, dx$$

This integral seems complicated, but if we take a closer look, we can see that there is a lot of room for simplification. First, because the rate of decay of an exponential function is faster than the rate of increase of a first-order polynomial, the first term converges to zero. Therefore, we have

$$\sigma^2 = \frac{1}{k} \sqrt{\frac{k}{2 \pi}} \int_{- \infty}^\infty e^{- \frac{k}{2}(x - \mu)^2} \, dx \tag{6}$$

But since 

$$\sqrt{\frac{k}{2 \pi}} \int_{- \infty}^\infty e^{- \frac{k}{2}(x - \mu)^2} \, dx = \int_{- \infty}^\infty f(x) \, dx = 1$$

Therefore,

$$\sigma^2 = \frac{1}{k}$$

Great! Now we know what the constant $$k$$ is:

$$k = \frac{1}{\sigma^2}$$

Plugging this expression back into (5), we finally have the equation for the probability distribution function of the univariate Gaussian.

$$f = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac12 (\frac{x - \mu}{\sigma})^2} \tag{7}$$

And now we’re done!

## Critical Points Analysis

Let’s perform a quick sanity check on (7) by identifying its critical points. Based on prior knowledge, we would expect to find the local maximum at $$x = \mu$$, as this is where the bell curve peaks. If we were to dig a bit deeper into prior knowledge, we would expect the point of inflection to be one standard deviations away from the mean, left and right. Let’s verify if these are actually true.

### Local Maximum

From good old calculus, we know that we can obtain the local extrema by setting the first derivative to zero. 

$$\frac{df}{dx} = - \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac12 (\frac{x - \mu}{\sigma})^2} \frac{1}{\sigma^2} (x - \mu) = 0$$

We can ignore the constants as they are non-zero. Then, we end up with

$$\frac{df}{dx} \propto (x - \mu) e^{- \frac12 (\frac{x - \mu}{\sigma})^2} = 0$$

Because the exponent is always positive, the only way for the expression to evaluate to zero is if 

$$x = \mu$$

This tells us that the local maximum of the univariate Gaussian occurs at the mean of the distribution, as we expect.

### Inflection Points

The inflection point can be obtained by setting the second order derivative of the probability distribution function equal to zero. Luckily, we’re already halfway done with calculating the second order derivative since we’ve already computed the first order derivative above. As we have done above, let’s ignore the constants since they don’t affect the calculation.

$$\frac{d^2f}{dx^2} \propto \frac{d}{dx}(x - \mu) e^{- \frac12 (\frac{x - \mu}{\sigma})^2} = e^{- \frac12 (\frac{x - \mu}{\sigma})^2} - \frac{x - \mu}{\sigma^2}e^{- \frac12 (\frac{x - \mu}{\sigma})^2}(x - \mu) = 0$$

Because the first exponential term cannot equal zero, we can simplify the equation to

$$(1 - \frac{(x - \mu)^2}{\sigma^2}) = 0$$

Therefore, 

$$\frac{x - \mu}{\sigma} = \pm 1$$

$$x = \mu \pm \sigma$$

From this, we can see that the inflection point of the univariate Gaussian is exactly one standard deviation away from the mean. This is one of the many interesting properties of the normal distribution that we can see from the formula for the probability distribution. 

# Multivariate Gaussian

So far, we’ve looked at the univariate Gaussian, which involved only one random variable $$X = x$$. However, what if the random variable in question is a vector that contains multiple random variables? It is not difficult to see that answering this question requires us to think in terms of matrices, which is the go-to method of packaging multiple numbers into neat boxes, known as matrices. 

Instead of deriving the probability distribution for the multivariate Gaussian from scratch as we did for the univariate case, we’ll build on top of the equation for the univariate Gaussian to provide an intuitive explanation for the multivariate case. 

## The Scalar-Matrix Parallel

In a previous post on linear regression, we took a look at matrix calculus to cover basic concepts such as the gradient. We established some rough intuition by associating various matrix calculus operations and their single-variable calculus analogues. Let’s try to use this intuition as a pivot point to extend the univariate Gaussian model to the multivariate Gaussian. 

For readability sake, here is the univariate model we have derived earlier.

$$f = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac12 (\frac{x - \mu}{\sigma})^2} \tag{7}$$

Examining (7), the first observation we might make is that $$(x - \mu)^2$$ is no longer a coherent expression in the multivariable context. The fix to this is extremely simple: recall that 

$$a^2 = a^{T}a$$
in vector world. Therefore, we can reexpress (7) as

$$f = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{1}{2 \sigma^2} (x - \mu)^{T}(x - \mu)}$$

This is the result of simply changing the squared term. Continuing, the next subject of our interest would be $$\sigma$$, as the variance is only strictly defined for one variable, as expressed by its definition below:

$$\sigma^2 = \mathbf{E}(X^2) - \mathbf{E}(X)\mathbf{E}(X)$$

Here, $$x$$ is a random variable, which takes a scalar value. The multivariable analogue of variance is covariance, which is defined as

$$Cov(X, Y) = $$










[this post]: https://jaketae.github.io/study/likelihood/


