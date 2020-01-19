---
title: Exponential: The Family Name of Distributions
mathjax: true
toc: true
categories:
  - study
tags:
  - statistics
  - probability_distribution
---

Definition

$$\{p_\theta: \theta \in \Theta \}$$

$$p_\theta (x) = \frac{ h(x) \ \text{exp}\left(- \sum_{i = 1}^m \eta_i(\theta) s_i(x) \right)}{z(\theta)} $$

$$x \in \mathbb{R}^d, \ \theta \in \mathbb{R}^k, \ \eta_i: \Theta \to \mathbb{R}, \ s_i: \mathbb{R}^d \to \mathbb{R}, \ h: \mathbb{R}^d \to [0, \infty), \ z: \Theta \to (0, \infty)$$

$$\eta(\theta) = \begin{pmatrix} \eta_1(\theta) \\ \eta_2(\theta) \\ \vdots \\ \eta_m(\theta) \end{pmatrix}, \ s(x) = \begin{pmatrix} s_1(x) \\ s_2(x) \\ \vdots \\ s_m(x) \end{pmatrix}$$

$$p_\theta (x) = \frac{h(x) \ \text{exp}\left(\eta(\theta)^T s(x) \right)}{z(\theta)} $$

Exponential Distribution

$$p_\theta(x) = \theta e^{- \theta x} I(x)$$

$$I(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}$$

$$\eta(\theta) = \theta, \ s(x) = -x, \ h(x) = I(x), \ z(\theta) = \frac{1}{\theta}$$

Bernouli Distribution

$$p_\theta(x) = \theta^x(1 - \theta)^{1 - x}I(x)$$

$$I(x) = \begin{cases} 1 & x \in \{0, 1\} \\ 0 & x \not \in \{0, 1\} \end{cases}$$

$$e^{\log(p_\theta(x))} = p_\theta(x) = I(x) \ \text{exp}\left(x\log(\theta) + (1 - x)\log(1 - \theta)\right)$$

$$\eta(\theta) = \begin{pmatrix} \log(\theta) & \log(1 - \theta) \end{pmatrix}^T, \ s(x) = \begin{pmatrix} x & 1 - x \end{pmatrix}^T, \ h(x) = I(x), \ z(\theta) = 1$$

Continuous RV Distributions in Exponential Family
* Exponential
* Gaussian
* Beta
* Gamma
* Chi-squared

Discrete RV Distributions in Exponential Family
* Bernoulli
* Binomial
* Poisson
* Geometric
* Multinomial

MLE Estimation with Canonical Form

$$\eta(\theta) = \theta$$

$$p_\theta(x) = \frac{h(x) \ \text{exp}\left(\theta^T s(x)\right)}{z(\theta)}$$

$$D = (x_1, x_2, \dots, x_n), \ x \in \mathbb{R}^d$$

$$\theta_{MLE} = \mathop{\rm arg\,max}\limits_{\theta \in \Theta} p(D \vert \theta)$$

$$\begin{align} p(D \vert \theta) &= \prod_{i = 1}^n p(x_i \vert \theta) \\ &= \frac{\prod_{i = 1}^n h(x_i) \ \text{exp}\left(\theta^T \left(\sum_{i = 1}^n s(x_i) \right) \right)}{z(\theta)^n} \\ &= \frac{\prod_{i = 1}^n h(x_i) \ \text{exp}\left(\theta^T s(D) \right)}{z(\theta)^n} \end{align}$$

$$\theta_{MLE} = \mathop{\rm arg\,max}\limits_{\theta \in \Theta} \log(p(D \vert \theta))$$

$$\log(p(D \vert \theta)) = -n\log(z(\theta)) + \theta^T s(D) + \sum_{i = 1}^n \log(h(x_i))$$

$$\frac{\partial \log(p(D \vert \theta))}{\partial \theta_j} = - n \frac{\partial \log(z(\theta))}{\partial \theta_j} + s_j(D)$$

$$\int_{X \in \mathbb{R}^d} p_\theta(x) = \int \frac{h(x) \ \text{exp}\left(\theta^T s(x)\right)}{z(\theta)} \, dx = 1$$

$$\therefore z(\theta) = \int h(x) \ \text{exp}\left(\theta^T s(x)\right) \, dx$$

$$\begin{align} \frac{\partial \log(z(\theta))}{\partial \theta_j} &= \frac{1}{z(\theta)} \int \frac{\partial \left(h(x) \ \text{exp}\left(\theta^T s(x)\right) \right)}{\partial \theta_j} \, dx \\ &= \frac{1}{z(\theta)} \int h(x) s_j(x) \ \text{exp}\left(\theta^T s(x) \right) \, dx \\ &= \int  s_j(x) \frac{h(x) \ \text{exp}\left(\theta^T s(x) \right)}{z(\theta)} \, dx \\ &= \int s_j(x) p(x \vert \theta) \, dx \\ &= \mathbb{E}_\theta \left[s_j(X)\right] \end{align}$$

$$\therefore \nabla_\theta \log(z(\theta)) = \mathbb{E}_\theta\left[s(X)\right]$$ 

$$\begin{align} \nabla_\theta \log(p(D \vert \theta)) &= - n \nabla_\theta \log(z(\theta)) + s(D) \\ &= - n \ \mathbb{E}_\theta \left[s(X)\right] + s(D) \\ &= 0 \end{align}$$

$$\mathbb{E}_\theta \left[s(X)\right] = \frac{s(D)}{n} = \frac{\sum_{i = 1}^n s(x_i)}{n}$$

Exponential Distribution

$$p_\theta(x) = \theta e^{- \theta x} I(x)$$

$$I(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}$$

$$\eta(\theta) = \theta, \ s(x) = -x, \ h(x) = I(x), \ z(\theta) = \frac{1}{\theta}$$

$$\begin{align} \nabla_\theta \log(z(\theta)) &= \frac{d}{d\theta}\left[\log\left(\frac{1}{\theta}\right) \right] \\ &= \frac{d}{d\theta}\left[- \log(\theta) \right] \\ &= - \frac{1}{\theta} \end{align}$$ 

$$\mathbb{E}_\theta \left[s(X)\right] = \mathbb{E}_\theta [-X] = - \frac{1}{\theta} \implies \mathbb{E}_\theta [X] = \frac{1}{\theta}$$

$$\nabla_\theta \log(z(\theta) = - \frac{1}{\theta} = \frac{\sum_{i = 1}^n -x_i}{n}$$

$$\therefore \theta_{MLE} = \frac{1}{\frac{1}{n} \sum_{i = 1}^n x_i}$$
