---
title: Dissecting the Gaussian Distribution
mathjax: true
date: 2019-12-12
toc: true
categories:
  - study
tags:
  - probability_distribution
  - statistics
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

$$ \frac{2A^2}{k} \left(\int_{- \infty}^{\infty} e^{-u^2} \, du \right) \left(\int_{- \infty}^{\infty} e^{-u^2} \, du \right) = 1$$

Because the two integrals are independent, *i.e.* calculating one does not impact the other, we can use two different variables for each integral. For notational convenience, let’s use $$x$$ and $$y$$.

$$ \frac{2A^2}{k} \left(\int_{- \infty}^{\infty} e^{-x^2} \, dx \right) \left(\int_{- \infty}^{\infty} e^{-y^2} \, dy \right) = 1$$

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

## Derivation Through Scalar-Matrix Parallel

In a previous post on linear regression, we took a look at matrix calculus to cover basic concepts such as the gradient. We established some rough intuition by associating various matrix calculus operations and their single-variable calculus analogues. Let’s try to use this intuition as a pivot point to extend the univariate Gaussian model to the multivariate Gaussian. 

For readability sake, here is the univariate model we have derived earlier.

$$f = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac12 (\frac{x - \mu}{\sigma})^2} \tag{7}$$

Examining (7), the first observation we might make is that $$(x - \mu)^2$$ is no longer a coherent expression in the multivariable context. The fix to this is extremely simple: recall that 

$$a^2 = a^{T}a$$

in vector world. Therefore, we can reexpress (7) as

$$f = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{1}{2 \sigma^2} (x - \mu)^{T}(x - \mu)}$$

This is the result of simply changing the squared term. Continuing, the next subject of our interest would be $$\sigma$$, as the [variance] is only strictly defined for one variable, as expressed by its definition below:

$$\sigma^2 = \mathbf{E}(X^2) - \mathbf{E}(X)\mathbf{E}(X)$$

where $$X$$ is a random variable, which takes a scalar value. This necessarily begs the question: what is the multivariable equivalent of variance? To answer this question, we need to understand covariance and the covariance matrix.

### Covariance

To jump right into the answer, the multivariable analogue of variance is [covariance], which is defined as

$$\text{Cov}(X, Y) = \mathbf{E}(X - \mu_X)\mathbf{E}(Y - \mu_Y) = \mathbf{E}(XY) - \mu_X \mu_Y \tag{8}$$

Notice that $$\text{Cov}(X, X)$$ equals variance, which is why we stated earlier that covariance is the multivariate equivalent of variance for univariate quantities. 

The intuition we can develop from looking at the equation is that covariance measures how far our random variables are from the mean in the $$X$$ and $$Y$$ directions. More concretely, covariance is expresses the degree of association between two variables. Simply put, if there is a positive relationship between two variables, *i.e.* an increase in one variable results in a corresponding increase in the other, the variance will be positive; conversely, if an increase in one variable results in a decrease in the other, covariance will be negative. A covariance of zero signifies that there is no linear relationship between the two variables. At a glance, the concept of covariance bears strong resemblance to the notion of [correlation], which also explains the relationship between two variables. Indeed, covariance and correlation are related: in fact, correlation is a function of covariance. 

$$\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

The biggest difference between correlation and covariance is that correlation is bounded between -1 and 1, whereas covariance is unbounded. The bottom line is that both correlation and covariance measure the strength of linearity between two variables, with correlation being a normalized version of covariance.

At this point in time, one might point out that covariance is not really a multivariate concept since it is defined for only two variables, not three or more. Indeed, the expression $$\text{Cov}(X, Y, Z)$$ is mathematically incoherent. However, covariance can be  a multivariate metric since we can express the covariance of any pairs of random variables by constructing what is called the [covariance matrix].

### Covariance Matrix

Simply put, the covariance matrix is a matrix whose elements are the pairwise covariance of two random variables in a random vector. Before we get into the explanation, let's take a look at the equation for the covariance matrix:

$$\Sigma = \mathbf{E}((X - \mathbf{E}(X))(X - \mathbf{E}(X))^{T}) \tag{9}$$

where $$\Sigma \in \mathbb{R}^{n \times n}$$ and $$\mathbf{E}(X) = \mu \in \mathbb{R}^{n \times 1}$$. This is the matrix analogue of the expression

$$\sigma^2 = \mathbf{E}((X - \mu)^2)$$

which is an alternate definition of variance. It is natural to wonder why we replaced the squared expression with $$(X - \mu)(X - \mu)^{T}$$ instead of $$(X - \mu)^{T}(X - \mu)$$ as we did earlier with the term in the exponent. The simplest answer that covariance is expressed as a matrix, not a scalar value. By dimensionality, $$(X - \mu)(X - \mu)^{T}$$ produces a single scalar value, whereas $$(X - \mu)(X - \mu)^{T}$$ creates a matrix of rank one. We can also see why (9) is coherent by unpacking the expected values expression as shown below:

$$\mathbf{E}((X - \mathbf{E}(X))(X - \mathbf{E}(X))^{T}) = \mathbf{E}(XX^T - X \mathbf{E}(X)^T - \mathbf{E}(X)X^T + \mathbf{E}(X)\mathbf{E}(X)^T)$$

Using the linearity of expectation, we can rewrite the equation as

$$\mathbf{E}(XX^T) - \mathbf{E}(X)\mathbf{E}(X)^T - \mathbf{E}(X)\mathbf{E}(X)^T + \mathbf{E}(X)\mathbf{E}(X)^T$$

Therefore, we end up with

$$\Sigma = \mathbf{E}(XX^T) - \mathbf{E}(X)\mathbf{E}(X)^T$$

which almost exactly parallels the definition of variance, which we might recall is 

$$\sigma^2 = \mathbf{E}(X^2) - \mathbf{E}(X)\mathbf{E}(X)$$

where $$\mu = \mathbf{E}(X)$$. The key takeaway is that the covariance matrix constructed from the random vector $$X$$ is the multivariable analogue of variance, which is a function of the random variable $$x$$. To gain a better idea of what the covariance matrix actually looks like, however, it is necessary to review its structure element-by-element. Here is the brief sketch of the $$n$$-by-$$n$$ covariance matrix. 

$$\Sigma = \begin{pmatrix} (X_1 - \mathbf{E}(X_1))(X_1 - \mathbf{E}(X_1)) && \dots && (X_1 - \mathbf{E}(X_1))(X_n - \mathbf{E}(X_n)) \\ \vdots && \ddots && \vdots \\ (X_n - \mathbf{E}(X_n))(X_1 - \mathbf{E}(X_1)) && \dots && (X_n - \mathbf{E}(X_n))(X_n - \mathbf{E}(X_n)) \end{pmatrix}$$

This might seem complicated, but using the definition of covariance in (8), we can simplify the expression as:

$$\Sigma = \begin{pmatrix} \text{Cov}(X_1, X_1) && \dots && \text{Cov}(X_1, X_K) \\ \vdots && \ddots && \vdots \\ \text{Cov}(X_K, X_1) && \dots && \text{Cov}(X_K, X_K) \end{pmatrix} \tag{10}$$

Note that the covariance matrix is a symmetric matrix since $$\Sigma = \Sigma^{T}$$. More specifically, the covariance matrix is a [positive semi-definite matrix]. This flows from the definition of positive semi-definiteness. Let $$u$$ be some arbitrary non-zero vector. Then,

$$u^T \Sigma u = u^T \mathbf{E}((X - \mu)(X - \mu)^{T}) u = \mathbf{E}(u^T (X - \mu)(X - \mu)^{T} u) = \mathbf{E}(\lvert (X - \mu)^{T} u \rvert^2) \geq 0$$

You might be wondering how (9) ends up as (10). Although this relationship may not be immediately apparent, that the two expressions are identical can be seen by setting the random vector as 

$$X = \begin{pmatrix} X_1 \\ X_2 \\ \vdots \\ X_K \end{pmatrix}$$

and performing basic matrix vector multiplication operations. For the sake of brevity, this is left as an exercise for the reader. 

### Putting Everything Together

We now have all the pieces we need to complete the puzzle. Recall that we were trying to derive the probability density function of the multivariate Gaussian by building on top of the formula for the univariate Gaussian distribution. We finished at

$$f = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{1}{2 \sigma^2} (x - \mu)^{T}(x - \mu)}$$

then moved onto a discussion of variance and covariance. Now that we understand that the covariance matrix is the analogue of variance, we can substitute $$\sigma^2$$ with $$\Sigma$$, the covariate matrix. 

$$f = \frac{1}{\sqrt{2 \pi \Sigma}} e^{- \frac{1}{2 \Sigma} (x - \mu)^{T}(x - \mu)}$$

Instead of leaving $$\Sigma$$ at the denominator, let's use the fact that 

$$\frac{1}{\Sigma} = \Sigma^{-1}$$

to rearrange the expression. This is another example of when the matrix-scalar parallel intuition can come in handy: the scalar multiplicative identity is 1, whereas the equivalent in matrix world is the identity matrix $$I$$. Therefore, the reciprocal of a matrix can be interpreted as its inverse. From this observation, we can conclude that

$$f = \frac{1}{\sqrt{2 \pi \Sigma}} e^{- \frac12 (x - \mu)^{T}\Sigma^{-1}(x - \mu)}$$

We are almost done, but not quite. Recall the the constant coefficient of the probability distribution originates from the fact that 

$$\int_{- \infty}^{\infty} f = 1$$

We have to make some adjustments to the constant coefficient since, in the context of the multivariate Gaussian, the integral translates into

$$\int_{x \in \mathbb{R}^n} f(x; \mu, \Sigma) \, dx = \int_{- \infty}^{\infty} \cdots \int_{- \infty}^{\infty} f(x; \mu, \Sigma) \, dx_1 \dots \, dx_n$$

While it may not be apparent immediately, it is not hard to accept that the correcting coefficient in this case has to be 

$$\frac{1}{\sqrt{(2 \pi)^n} \lvert \Sigma \rvert}$$ 

as there are $$n$$ layers of iterated integrals to evaluate for each $$x_1$$ through $$x_n$$. Instead of the matrix $$\Sigma$$, we use its determinant $$\lvert \Sigma \rvert$$ since we need the coefficient to be a constant, not a matrix term. We don't go into much detail about the derivation of the constant term; the bottom line is that we want the integral of the probability distribution function over the relevant domain to converge to 1. 

If we put the pieces of the puzzle back together, we finally have the probability distribution of the multivariate Gaussian distribution:

$$f = \frac{1}{\sqrt{(2 \pi)^n \lvert \Sigma \rvert}} e^{- \frac12 (x - \mu)^{T}\Sigma^{-1}(x - \mu)} \tag{11}$$

## Example with Diagonal Covariance Matrix

To develop a better intuition for the multivariate Gaussian, let's take a look at a case of a simple 2-dimensional Gaussian random vector with a diagonal covariance matrix. This example was borrowed from [this source].

$$\begin{align*} x = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}, && \mu = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, && \Sigma = \begin{pmatrix} \sigma_1 & 0 \\ 0 & \sigma_2 \end{pmatrix} \end{align*}$$

Using the formula for the multivariate Gaussian we derived in (11), we can construct the probability distribution function given $$X$$, $$\mu$$, and $$\Sigma$$. 

$$f(x; \mu, \Sigma) = \frac{1}{2 \pi ({\sigma_1}^2 \cdot {\sigma_2}^2)^{\frac12}} \text{exp} \left(- \frac12 \begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix}^T \begin{pmatrix} \frac{1}{\sigma_1^2} & 0 \\ 0 & \frac{1}{\sigma_2^2} \end{pmatrix} \begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix} \right)$$

Note that computing $$\Sigma^{-1}$$, the inverse of the covariance matrix, can be accomplished simply by taking the reciprocal of its diagonal entries since $$\Sigma$$ was assumed to be a diagonal matrix. Continuing, 

$$\begin{align*} f(x; \mu, \Sigma) &= \frac{1}{2 \pi \sigma_1 \sigma_2} \text{exp} \left(- \frac12 \begin{pmatrix} x_1 - \mu_1 \\ x_2 - \mu_2 \end{pmatrix}^T \begin{pmatrix} \frac{1}{\sigma_1^2}(x_1 - \mu_1) \\ \frac{1}{\sigma_2^2}(x_2 - \mu_2) \end{pmatrix} \right) \\ &= \frac{1}{2 \pi \sigma_1 \sigma_2} \text{exp} \left(- \frac{1}{2 \sigma_1^2} (x_1 - \mu_1)^2 - \frac{1}{2 \sigma_2^2} (x_2 - \mu_2)^2 \right) \\ &= \frac{1}{\sqrt{2 \pi} \sigma_1} \text{exp} \left( - \frac{1}{2 \sigma_1^2} (x_1 - \mu_1)^2 \right) \cdot \frac{1}{\sqrt{2 \pi} \sigma_2} \text{exp} \left( - \frac{1}{2 \sigma_2^2} (x_2 - \mu_2)^2 \right) \end{align*}$$

In other words, the probability distribution of seeing a random vector $$\begin{pmatrix} x_1 & x_2 \end{pmatrix}^T$$ given $$x_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$$ and $$x_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$$ is equal to the product of the two univariate Gaussians. This result is what we would expect given that $$\text{Cov}(x_1, x_2) = 0$$. For instance, if $$x_1$$ and $$x_2$$ are independent, *i.e.* observing a value of $$x_1$$ does not inform us of anything about $$x_2$$ and vice versa, it would make sense that the possibility of observing a random vector $$x$$ with entries $$x_1$$ and $$x_2$$ is merely the product of the independent probabilities of each observing $$x_1$$ and $$x_2$$. This example illustrates the intuitive link between the multivariate and univariate Gaussian distributions. 

# Conclusion

In this post, we took a look at the normal distribution from the perspective of probability distributions. By working from the definition of what constitutes a normal data set, we were able to completely build the probability density function from scratch. The derivation of the multivariate Gaussian was complicated by the fact that we were dealing with matrices and vectors instead of single scalar values, but the matrix-scalar parallel intuition helped us a lot on the way. Note that the derivation of the multivariate Gaussian distribution introduced in this post is not a rigorous mathematical proof, but rather intended as a gentle introduction to the multivariate Gaussian distribution. 

I hope you enjoyed reading this post on normal distributions. Catch you up in the next one.


[this post]: https://jaketae.github.io/study/likelihood/
[previous post]: https://jaketae.github.io/study/svd/
[variance]: https://en.wikipedia.org/wiki/Variance
[covariance]: https://en.wikipedia.org/wiki/Covariance
[covariance matrix]: https://en.wikipedia.org/wiki/Covariance_matrix
[correlation]: https://en.wikipedia.org/wiki/Correlation_and_dependence
[positive semi-definite matrix]: https://en.wikipedia.org/wiki/Definiteness_of_a_matrix
[this source]: http://cs229.stanford.edu/section/gaussians.pdf