---
title: Convex Combinations and MAP
mathjax: true
toc: true
categories:
  - study
tags:
  - statistics

---

MAP for mean of univariate Gaussian

$$D = (x_1, x_2, \dots, x_n), \ x \in \mathbb{R}$$

$$\theta \sim \mathcal{N}(\mu, 1)$$

$$X \sim \mathcal{N}(\theta, \sigma^2)$$

$$p(D \vert \theta) = p(x_1, x_2, \dots, x_n \vert \theta) = \prod_{i = 1}^n p(x_i \vert \theta) $$

$$\begin{align}\theta_{MAP} &= \mathop{\rm arg\,max}\limits_{\theta} p(\theta \vert D)\\ &= \mathop{\rm arg\,max}\limits_{\theta} \frac{p(D \vert \theta) p(\theta)}{p(D)} \\ &= \mathop{\rm arg\,max}\limits_{\theta} p(D \vert \theta) p(\theta) \\ &= \mathop{\rm arg\,max}\limits_{\theta} \log(p(D \vert \theta) p(\theta)) \\ &= \mathop{\rm arg\,max}\limits_{\theta} \log(p(D \vert \theta)) + \log(p(\theta)) \end{align}$$

$$p(x \vert \theta) = \frac{1}{\sigma \sqrt{2 \pi}} \text{exp}\left({\frac{(x - \theta)^2}{2\sigma^2}}\right)$$

$$\log(p(D \vert \theta)) = \log(\prod_{i = 1}^n p(x_i \vert \theta)) = \sum_{i = 1}^n \log(p(x_i \vert \theta)) = n\log(\frac{1}{\sigma \sqrt{2 \pi}}) + \text{exp} \left(\frac{1}{2\sigma^2}\sum_{i = 1}^n (x_i - \theta)^2 \right)$$

$$p(\theta) = \frac{1}{\sqrt{2 \pi}} \text{exp}\left(\frac{1}{2}(\theta - \mu)^2 \right)$$

$$\log(p(\theta)) = \log \left(\frac{1}{\sqrt{2 \pi}}\right) + \frac{\theta^2}{2}$$

$$\begin{align} \frac{d}{d \theta}\left[ \log(p(D \vert \theta)) + \log(p(\theta)) \right] &= \frac{d}{d \theta}\left[ \log(p(D \vert \theta)) \right] + \frac{d}{d \theta}\left[\log(p(\theta)) \right] \\ &= \frac{1}{\sigma^2} \sum_{i = 1}^n (x_i - n \theta) + ... = 0 \end{align}$$





