---
title: Leibniz or Feynman?
mathjax: true
toc: true
categories:
  - study
tags:
  - 
---

Before I begin, I must say that [this video](https://www.youtube.com/watch?v=zbWihK9ibhc) by Brian Storey at Olin College is the most intuitive explanation of the Leibniz rule I have seen so far. Granted, my greedy search over the internet space was by no means exhaustive, so I've probably missed some other hidden gems  here and there. Also, the video is intended as a visual explanation for beginners rather than a robust analytical proof of the Leibniz rule. This point notwithstanding, I highly recommend that you check out the video.

This post is going to provide a short, condensed summary of the proof presented in the video, minus the fancy visualization that pen and paper can afford. 

The Leibniz rule, sometimes referred to as Feynman's rule or differentiation-under-the-integral-sign-rule, is an interesting, highly useful way of computing complicated integrals. A simple version of the Leibniz rule might be stated as follows:


$$
\frac{d}{dt} \int_{a}^b f(x, t) \, dx = \int_{a}^b \frac{d}{dt}f(x, t) \, dx
$$


As you can see, what this rule essentially tells us is that integrals and derivatives are interchangeable under mild conditions. We've used this rule many times in a previous post on [Fisher's information matrix](https://jaketae.github.io/study/fisher/) when computing expected values that involved derivatives. 

Why is this the case? It turns out that the Leibniz rule can be proved by using the definition of derivatives and some Taylor expansion. Recall that the definition of a derivative can be written as


$$
\frac{df}{dx} = \lim_{h \to 0}\frac{f(x + h) - f(x)}{h}
$$


This is something that we'd see straight out of a calculus textbook. As simple as it seems, we can in fact analyze Leibniz's rule by applying this definition, as shown below:


$$
\begin{align}
\frac{d}{dt} \int_{a}^b f(x, t) \, dx 
&= \lim_{h \to 0}\frac{\int_{a}^b f(x, t + h) \, dx - \int_{a}^b f(x, t) \, dx}{h} \\
&= \lim_{h \to 0}\frac{\int_{a}^b f + h \frac{\partial f}{\partial t}  \, dx - \int_{a}^b f \, dx}{h} \\ 
&= \lim_{h \to 0}\frac{\int_{a}^b h \frac{\partial f}{\partial t} dx}{h} \\
&= \int_{a}^b \frac{\partial f}{\partial t} \, dx \\
\end{align}
$$


Thus we have shown that, if the limits of integration are constants, we can switch the order of integration and differentiation. 

But because our quench for knowledge is insatiable, let's consider the more general case as well: when the limits are not bounded by constant, but rather functions. Specifically, the case we will consider looks as follows.


$$
\frac{d}{dt} \int_{a(t)}^{b(t)} f(x, t) \, dx 
$$


In this case, we see that $a$ and $b$ are each functions of variable $t$. With some thinking, it is not difficult to convince ourselves that this will indeed introduce some complications that require modifications to our original analysis. 











