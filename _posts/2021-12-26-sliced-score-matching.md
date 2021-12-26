---
title: Score Matching
mathjax: true
toc: fa
categories:
  - study
tags:
  - deep_learning
  - statistics
---

Recently, I've heard a lot about score-based networks. In this post, I will attempt to provide a high-level overview of what scores are and how the concept of score matching gives rise to a family of likelihood-based generative models. This post is heavily adapted from [Yang Song's post on sliced score matching](https://yang-song.github.io/blog/2019/ssm/).

# Probability Model

Given a parametrized real-valued function $f_\theta(\mathbf{x})$, we can derive a probability model $p_\theta(\mathbf{x})$ by applying a normalization term $Z_\theta$.

$$
p_\theta (\mathbf{x}) = \frac{e^{- f_\theta (\mathbf{x})}}{Z_\theta} \\
Z_\theta = \int e^{- f_\theta (\mathbf{x})} \, d \mathbf{x}.
$$

In practice, $f_\theta$ is often an energy-based model (EBM).

We can then define the likelihood function as follows:

$$
\log p_\theta (\mathbf{x}) = - f_\theta (\mathbf{x}) - \log Z_\theta.
$$

However, one glaring problem with this formulation is that $Z_\theta$ is often intractable. Score-matching presents an elegant solution to bypass this problem.

# Score-Matching

To eliminate the intractable term, we consider the score, which is defined as the gradient of the log likelihood with respect to the random variable $\mathbf{x}$. Note that we are not taking the gradient with respect to the parameter $\theta$, which is typically the object of interest in processes such as MLE.

$$
\nabla_\mathbf{x} \log p_\theta (\mathbf{x}) = - \nabla_\mathbf{x} f_\theta (\mathbf{x}).
$$

The goal of score-matching, then, is to minimize the difference between $p_\text{data}$ and $p_\theta$ by optimizing the Fisher divergence. For sake of simplicity, we consider the 1-D case.

$$
\begin{align}
&\frac12 \mathbb{E}_{p_\text{data}} \lVert \nabla_x \log p_\text{data} (x) - \nabla_x \log p_\theta (x) \rVert^2_2 \\
&= \frac12 \int p_\text{data} (x) \left( \nabla_x \log p_\text{data} (x) - \nabla_x \log p_\theta (x) \right)^2 \, dx \\
&= \frac12 \int p_\text{data}(x) (\nabla_x \log p_\text{data}(x))^2 \, dx + \frac12 \int p_\text{data} (x) (\nabla_x \log p_\theta (x))^2 \, dx \\
& - \int p_\text{data}(x) \nabla_x \log p_\text{data}(x) \nabla_x \log p_\theta (x) \, dx .
\end{align}
$$

The equalities simply follow from the integral definition of expectation. Note that the first term is simply a constant and can be ignored during optimization.

Applying integration by parts on the last term,

$$
\begin{align}
& \int p_\text{data}(x) \nabla_x \log p_\text{data}(x) \nabla_x \log p_\theta (x) \, dx \\
&= \int p_\text{data}(x) \frac{\nabla_x p_\text{data}(x)}{p_\text{data} (x)} \nabla_x \log p_\theta (x) \, dx \\
&= \int \nabla_x \log p_\theta (x) \nabla_x p_\text{data} (x) \, dx \\
&= p_\text{data}(x) \nabla_x \log p_\theta(x) \bigg|^\infty_{- \infty} - \int p_\text{data}(x) \nabla^2_x \log p_\theta (x) \, dx \\
& \approx - \mathbb{E}_{p_\text{data}}[\nabla^2_x \log p_\theta (x)].
\end{align}
$$

Putting all terms together,

$$
\begin{align}
&\frac12 \mathbb{E}_{p_\text{data}} \lVert \nabla_x \log p_\text{data} (x) - \nabla_x \log p_\theta (x) \rVert^2_2 \\
&= \mathbb{E}_{p_\text{data}}[\nabla^2_x \log p_\theta (x)] + \frac12 \mathbb{E}_{p_\text{data}} [(\nabla_x \log p_\theta (x))^2] + \text{const.} \\
&= \mathbb{E}_{p_\text{data}}[\nabla^2_x \log p_\theta (x) + \frac12 (\nabla_x \log p_\theta (x))^2] + \text{const.}
\end{align}
$$

We can easily extend this into a multidimensional context, the result of which is

$$
\mathbb{E}_{p_\text{data}} \left[\text{tr}(\nabla^2_\mathbf{x} \log p_\theta (\mathbf{x})) + \frac12 \lVert \nabla_\mathbf{x} \log p_\theta (\mathbf{x}) \rVert^2_2 \right] + \text{const.}
$$

# Sliced Score-Matching

We are specifically interested in instances where $f_\theta$ is parametrized as a neural network. Recall that

$$
\nabla_\mathbf{x} \log p_\theta (\mathbf{x}) = - \nabla_\mathbf{x} f_\theta (\mathbf{x}).
$$

Therefore, we can rewrite the score-matching objective as

$$
\mathbb{E}_{p_\text{data}} \left[\text{tr}(\nabla^2_\mathbf{x} f_\theta (\mathbf{x})) + \frac12 \lVert \nabla_\mathbf{x} f_\theta (\mathbf{x}) \rVert^2_2 \right] + \text{const}.
$$

While the first-order gradient can be simply obtained via backpropagation, $\text{tr}(\nabla^2_\mathbf{x} f_\theta (\mathbf{x}))$ is very computationally costly. To circumvent this problem, the authors propose random projection, which reduces dimensionality of data down to scalars. Quoting Yang Song:

> We propose **sliced score matching** to greatly scale up the computation of score matching. The motivating idea is that one dimensional data distribution is much easier to estimate for score matching. We propose to project the scores onto random directions, such that the vector fields of scores of the data and model distribution become scalar fields. We then compare the scalar fields to determine how far the model distribution is from the data distribution. It is clear to see that the two vector fields are equivalent if and only if their scalar fields corresponding to projections onto all directions are the same.

The random projection version of Fisher divergence is

$$
\frac{1}{2}\mathbb{E}_{p_\text{data}}[(\mathbf{v}^\intercal \nabla_\mathbf{x} \log p_\text{data}(\mathbf{x}) - \mathbf{v}^\intercal \nabla_\mathbf{x} \log p_\theta(\mathbf{x}) )^2].
$$

Intuitively, the equation forces the two distributions to get closer according to some random projection $\mathbf{v}$. Since the projection is random, there exists a guarantee that optimizing this quantity will bring $p_\theta$ closer to the real data distribution.

The sliced score-matching objective under this revised Fischer divergence is

$$
\mathbb{E}_{p_\text{data}}\bigg[\mathbf{v}^\intercal \nabla_{\mathbf{x}}^2 \log p_\theta(\mathbf{x})\mathbf{v} + \frac{1}{2} (\mathbf{v}^\intercal\nabla_\mathbf{x} \log p_\theta(\mathbf{x}))^2 \bigg] + \text{const}.
$$

The problem has now been reduced into computationally tractable form.

_This post was originally written in July, but polished into its current final form in December. If you spot any rough edges or details I missed, please feel free to reach out to me with corrections._
