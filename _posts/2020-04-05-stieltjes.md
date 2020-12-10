---
title: On Expectations and Integrals
mathjax: true
toc: false
categories:
  - study
tags:
  - statistics
---

Expectation is a core concept in statistics, and it is no surprise that any student interested in probability and statistics may have seen some expression like this:


$$
\mathbb{E}[X] = \sum_{x \in X} x f(x) \tag{1}
$$


In the continuous case, the expression is most commonly presented in textbooks as follows:


$$
\mathbb{E}[X] = \int_{x \in X} x f(x) \, dx \tag{2}
$$


However, this variant might throw you off, which happened to me when I first came across it a few weeks ago:


$$
\mathbb{E}[X] = \int_{x \in X} x \,dF(x) \tag{3}
$$


I mean, my calculus is rusty, but it kind of makes sense: the probably density function is, after all, a derivative of the cumulative density function, and so notationally there is some degree of coherency here.


$$
f(x) = \frac{d}{dx}F(x) \implies f(x) \, dx = dF(x) \tag{4}
$$


But still, this definition of the expected value threw me off quite a bit. What does it mean to integrate over a distribution function instead of a variable? After some research, however, the math gurus at [Stack Exchange](https://math.stackexchange.com/questions/380785/what-does-it-mean-to-integrate-with-respect-to-the-distribution-function) provided me with an answer. So here is a brief summary of my findings. 

The integral that we all know of is called the Riemann integral. The confusing integral is in fact a generalization of the Riemann integral, known as the Riemann-Stieltjes integral (don't ask me how to pronounce the name of the Dutch mathematician). There is an even more general interpretation of integrals called the Lebesgue integral, but we won't get into that here. 

First, let's take a look at the definition. The definition of the integral is actually a lot simpler than what one might imagine. Here, $c_i$ is a value that falls within the interval $[x_i, x_{i+1}]$. 


$$
\int_a^b f(x) \, dg(x) = \lim_{n \to \infty}\sum_{i=1}^n f(c_i)[g(x_{i+1}) - g(x_i)] \tag{5}
$$


In short, we divide the interval of integration $[a, b]$ into $n$ infinitesimal pieces. Imagine this process as being similar to what we learn in Calculus 101, where integrals are visualized as an infinite sum of skinny rectangles as the limit approaches zero. Essentially, we are doing the same thing, except that now, the base of each rectangle is defined as the difference between $g(x_{i+1})$ and $g(x_i)$ instead of $x_{i+1}$ and $x$ as is the case with the Riemann integral. Another way to look at this is to consider the integral as calculating the area beneath the curve represented by the parameterization $(x, y) = (g(x), f(x))$. This connection becomes a bit more apparent if we consider the fact that the Riemann integral is calculating the area beneath the curve represented by $(x, y) = (x, f(x))$. In other words, the Riemann-Stieltjes integral can be seen as dealing with a change of variables.

You might be wondering why the Riemann-Stieltjes integral is necessary in the first place. After all, the definition of expectation we already know by heart should be enough, shouldn't it? To answer this question, consider the following  function:


$$
F(x) = 
\begin{cases}
0 & x < 0\\\ 
\frac12 & 0 \leq x < 1 \\
1 & x \geq 1
\end{cases}

\tag{6}
$$


This cumulative mass function is obviously discontinuous since it is a step-wise function. This also means that it is not differentiable; hence, we cannot use the definition of expectation that we already know. However, this does not mean that the random variable $X$ does not have an expected value. In fact, it is possible to calculate the expectation using the Riemann-Stieltjes integral quite easily, despite the discontinuity!

The integral we wish to calculate is the following:


$$
\int_{-\infty}^{\infty} x \, dF(x) \tag{7}
$$


Therefore, we should immediately start visualizing splitting up the domain of integration, the real number line, into infinitesimal pieces. Each box will be of height $x$ and width $F(x_{i+1}) - F(x_i)$. In the context of the contrived example, this definition makes the calculation extremely easy, since  $F(x_{i+1}) - F(x_i)$ equals zero in all locations but the jumps where the discontinuities occur. In other words,


$$
\int_{-\infty}^{\infty} x \, dF(x) = 0 \cdot \frac12 + 1 \cdot 1 = 1 \tag{8}
$$


We can easily extend this idea to calculating things like variance or other higher moments. 

A more realistic example might be the [Dirac delta function](https://en.wikipedia.org/wiki/Dirac_delta_function). Consider a constant random variable (I know it sounds oxymoronic, but the idea is that the random variable takes only one value and that value only). In this case, we can imagine the probability density function as a literal spike in the sense that the PDF will peak at $x=c$ and be zero otherwise. The cumulative density function will thus exhibit a discontinuous jump from zero to 1 at $x=c$. And by the same line of logic, it is easy to see that the expected value of this random variable is $c$, as expected. Although this is a rather boring example in that the expectation of a constant is of course the constant itself, it nonetheless demonstrates the potential applications of Riemann-Stieltjes.

I hope you enjoyed reading this post. Lately, I have been busy working on some interesting projects. There is a lot of blogging and catching up to do, so stay posted for exciting updates to come!