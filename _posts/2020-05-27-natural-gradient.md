---
title: Natural Gradient and Fisher
mathjax: true
toc: true
categories:
  - study
tags:
  - statistics
  - machine_learning
---

In a previous post, we took a look at Fisher's information matrix. Today, we will be taking a break from the R frenzy and continue our exploration of this topic, meshing together related ideas such as gradient descent, KL divergence, Hessian, and more.

# Motivation

The typical formula for batch gradient descent looks something like this:


$$
\theta^{(k + 1)} = \theta^{(k)} - \alpha \nabla_\theta J(\theta) \tag{1}
$$


This is the familiar gradient descent algorithm that we know of. While this approach works and certainly makes sense, there are definite limitations; hence the introduction of other more efficient algorithms such as SGD, Adam, and et cetera. 

However, these algorithms all have one thing in common: they adjust the parameter in the parameter space according to Euclidean distance. In other words, gradient descent essentially looks at regions that are some Euclidean distance away from the current parameter and chooses the direction of steepest descent. 

This is where the notion of natural gradients come into play: if our goal is to minimize the cost function, which is effectively equivalent to maximizing the likelihood, why not search within the distribution space of the likelihood function instead? After all, this makes more sense since gradient descent in parameter space is likely to be easily perturbed by the mode of parametrization, such as using precision instead of variance in a normal distribution, whereas searching in the distribution space would not be subject to this limitation. So the alternative to this approach would be to search the distribution space and find the distribution that which makes value of the cost function the smallest. This is the motivation behind the notion of a natural gradient. 

# Divergence and Fisher

Now you might be wondering how all this has anything to do with the Fisher matrix, which we looked at in the previous post. Well, it turns out there are some deep, interesting questions to be posed and connections to be uncovered.

## Kullback-Leibler Divergence

If we're going to search around the distribution space, one natural question to consider is what distance metric we will use for our search. In case of batch gradient descent, we used Euclidean distance. This made sense since we were simply measuring the distance between two parameters, which are effectively scalars or vector quantities. If we want to search the distribution space, on the other hand, we would have to measure the distance between two probability distributions, one that is defined by the previous parameter and the other defined by the newly found parameter after natural gradient descent.

Well, we know one great candidate for this task right off the bat, and that is KL divergence. Recall that KL divergence is a way of quantifying the pseudo-distance between two probability distributions. The formula for KL divergence is shown below. And while we're at it, let's throw cross entropy and entropy into the picture as well, both for review and clarity's sake:


$$
\begin{align}
D_{KL}(p \parallel q) 
&= H(p, q) - H(p) \\
&= \int p(x) \log q(x) \, dx - \int p(x) \log q(x) \, dx \\
&= \int p(x) \log \left( \frac{q(x)}{p(x)} \right) \, dx
\end{align}
\tag{2}
$$


For a short, simple review of these concepts, refer to this [previous article](https://jaketae.github.io/study/information-entropy), or [Aurelien Geron's video on YouTube](https://www.youtube.com/watch?v=ErfnhcEV1O8).

In most cases, $p$ is the true distribution which we seek to model, while $q$ is some more tractable distribution at our disposal. In the classic context of ML, we want to minimize the KL divergence. In this case, however, we're simply using KL divergence as a means of measuring distance between two parameters in defined within a distribution space. As nicely stated in layman's term in this [Medium article](https://towardsdatascience.com/its-only-natural-an-excessively-deep-dive-into-natural-gradient-optimization-75d464b89dbb),  

> ... instead of “I’ll follow my current gradient, subject to keeping the parameter vector within epsilon distance of the current vector,” you’d instead say “I’ll follow my current gradient, subject to keeping the distribution my model is predicting within epsilon distance of the distribution it was previously predicting” 

I see this as an intuitive way of nicely summarizing why we're using KL divergence in searching the distribution space, as opposed to using Euclidean distance in searching the parameter space. 

## Fisher Matrix

Now it's time for us to connect the dots between KL divergence and Fisher's matrix. Before we diving right into computations, let's think about how or why these two concepts might be related at all. One somewhat obvious link is that both quantities deal with likelihood, or to be more precise, log likelihood. Due to the definition of entropy, KL divergence ends up having a log likelihood term, while Fisher's matrix is the negative expected Hessian of the log likelihood function, or the covariance matrix of Fisher's score, which is the gradient of the log likelihood. Either way, we know that likelihood is the fundamental bridge connecting the two.

Let's try to compute the KL divergence between $p(x \vert \theta)$ and $p(x \vert \theta')$. Conceptually, we can think of $\theta$ as the previous point of the parameter and $\theta'$ as the newly updated parameter. In this context, the KL divergence would tell us the effect of one iteration of natural gradient descent. This time, instead of using integral, let's try to simplify a bit by expressing quantities as expectations.


$$
D_{KL}[p(x \vert \theta) \parallel p(x \vert \theta')]= \mathbb{E}[\log p(x \vert \theta)] - \mathbb{E}[\log p(x \vert \theta')]
\tag{3}
$$


We see the familiar log likelihood term. Given the fact that the Fisher matrix is the negative expected Hessian of the log likelihood, we should be itching to derive this expression twice to get a Hessian out of it. Let's first obtain the gradient, then get its Jacobian to derive a Hessian. This derivation process was heavily referenced from [Agustinus Kristiadi's blog](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/).


$$
\begin{align}
\nabla_{\theta'} D_{KL}[p(x \vert \theta) \parallel p(x \vert \theta')]
&= \nabla_{\theta'} \mathbb{E}[\log p(x \vert \theta)] - \nabla_{\theta'} \mathbb{E}[\log p(x \vert \theta')] \\
&= - \nabla_{\theta'} \mathbb{E}[\log p(x \vert \theta')] \\
&= - \int p(x \vert \theta) \nabla_{\theta'}  \log p(x \vert \theta') \, dx 
\end{align}
\tag{4}
$$


Let's do this one more time to get the Hessian.


$$
\begin{align}
\nabla^2_{\theta'} D_{KL}[p(x \vert \theta) \parallel p(x \vert \theta')] 
&= - \nabla_{\theta'} \int p(x \vert \theta) \nabla_{\theta'}  \log p(x \vert \theta') \, dx \\ 
&= - \int p(x \vert \theta) \nabla^2_{\theta'}  \log p(x \vert \theta') \, dx \\
&= - \mathbb{E}[\nabla^2_{\theta'} \log p(x \vert \theta')] \\
&= - \mathbb{E}[\text{H}_{\log p(x \vert \theta')}] \\
&= \text{F}
\end{align}
\tag{5}
$$


This conclusion tells us that the curvature of KL divergence is defined by Fisher's matrix. In hindsight, this is not such a surprising result given that the KL divergence literally had a term for expected log likelihood. Applying the Leibniz rule twice to move the derivative into the integral, we quickly end up with Fisher's matrix.

# Natural Gradient

At this point, you might be wondering about the implications of this conclusion. It's great that KL divergence and the Fisher matrix are closely related via the Hessian, but what implication does it have for the gradient descent algorithm in distribution space? To answer this question, we first need to perform a quick multivariate second order Taylor expansion on KL divergence. 

## Taylor Expansion

Recall that the simple, generic case of multivariate Taylor expansion looks as follows:


$$
f(x) = f(x_0) + \nabla f(x_0)^\top f(x - x_0) + \frac12 (x - x_0)^\top \text{H} (x - x_0)
\tag{7}
$$


This is simply a generalization of the familiar univariate Taylor series approximation we saw earlier. (In most cases, we stop at the second order because computing the third order in the multivariate case requires us to obtain a three-dimensional symmetric tensor. I might write a post on this topic in the future, as I only recently figured this out and found it very amusing.) Continuing our discussion of KL divergence, let's try to expand the divergence term using Taylor approximation. Here, $\delta$ is small distance in the distribution space defined by KL divergence as the distance metric.


$$
D_{KL}[p_\theta \parallel p_{\theta + \delta}] \\
\approx D_{KL}[p_\theta \parallel p_{\theta'}]\|_{\theta' = \theta} + \nabla_{\theta'}D_{KL}[p_\theta \parallel p_\theta']\|_{\theta' = \theta} ^\top \delta + \frac12 \delta^\top \text{F}_{\theta'} \|_{\theta' = \theta} \delta \\
= \frac12 \delta^\top \text{F} \delta
\tag{8}
$$


This can be a bit obfuscating notation-wise because of the use of $\theta'$ as our variable, assuming $\theta$ as a fixed constant, and evaluating the gradient and the Hessian at the point where $\theta' = \theta$  since we want to approximate the value of KL divergence at the point where where $\theta' = \theta + \delta$. But really, all that is happening here is that in order to approximate KL divergence, we're starting at the point where $\theta' = \theta$, and using the slope and curvature obtained at that point to approximate the value of KL divergence at distance $\delta$ away. Picturing the simpler univariate situation in the Cartesian plane might help.

The bottom line is that the KL divergence is effectively defined by the Fisher matrix. The implication of this is that now, the gradient descent algorithm is subject to the constraint


$$
\delta^* = \mathop{\rm arg\,min}\limits_{\delta \text{ s. t. } D_{KL}[p_\theta \parallel p_{\theta + \delta}] = c} \mathcal{L}(\theta + \delta)
\tag{9}
$$


where $c$ is some constant. Now, the update rule would be


$$
\theta^{(k + 1)} = \theta^{(k)} + \delta^*
\tag{10}
$$


## Lagrangian

To solve for the argument minima operation, we will resort to the classic method for optimization: Lagrangians. In this case, the Lagrangian would be


$$
\delta^* = \mathop{\rm arg\,min}\limits_{\delta} \mathcal{L}(\theta + \delta) + \lambda(D_{KL}[p_\theta \parallel p_{\theta + \delta}] - c)
\tag{11}
$$


This immediately follows from using the constraint condition. 

To make progress, let's use Taylor approximation again, both on the term for the loss function and the KL divergence. The good news is that we have already derived the expression for the latter. 


$$
\delta^* \approx \mathop{\rm arg\,min}\limits_{\delta} \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}(\theta)^\top \delta + \lambda (\frac12 \delta^\top \text{F} \delta - c)
\tag{12}
$$


Noting the fact that there are several constants in this expression, we can simplify this into


$$
\delta^* \approx \mathop{\rm arg\,min}\limits_{\delta}  \nabla_\theta \mathcal{L}(\theta)^\top \delta + \frac12 \lambda \delta^\top \text{F} \delta
\tag{13}
$$


To minimize this expression, we set its gradient equal to zero. Note that we are deriving with respect to $\delta$.


$$
\frac{\partial}{\partial \delta} \left[ \nabla_\theta \mathcal{L}(\theta)^\top \delta + \frac12 \lambda \delta^\top \text{F} \delta \right] \\
= \nabla_\theta \mathcal{L}(\theta) + \lambda \text{F} \delta \\
= 0
\tag{14}
$$


Therefore, 


$$
\begin{align}
\delta^* 
&= - \frac{1}{\lambda} \text{F}^{-1} \nabla_\theta \mathcal{L}(\theta) \\
& \propto \text{F}^{-1} \nabla_\theta \mathcal{L}(\theta) \\
\end{align}
\tag{15}
$$


We are finally done with our derivation. This equation tells us that the direction of steepest descent is defined by the inverse of the Fisher matrix multiplied by the gradient of the loss function, up to some constant scaling factor. This is different from the vanilla batch gradient descent we are familiar with, which was simply defined as


$$
\delta^* = \nabla_\theta \mathcal{L}(\theta)
\tag{16}
$$


Although the difference seems very minor---after all, all that was changed was the addition of Fisher's matrix---yet the underlying concept, as we have seen in the derivation, is entirely different. 

# Conclusion

This was definitely a math-heavy post. Even after having written this entire post, I'm still not certain if I have understood the details and subtleties involved in the derivation. And even the details that I understand now will become confusing and ambiguous later when I return back to it. Hopefully I can retain most of what I have learned from this post. 

Before I close this post, I must give credit to [Agustinus Kristiadi](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/), whose blog post was basically the basis of this entire writing. I did look at a few Stack Overflow threads, but the vast majority of what I have written are either distillations or adaptations from their blog. It's a great resource for understanding the mathematics behind deep learning.

I hope you enjoyed reading this blog. See you in the next one!