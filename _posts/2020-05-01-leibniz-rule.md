---
title:  Understanding the  Leibniz Rule
mathjax: true
toc: false
categories:
  - study
tags:
  - analysis
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
&= \lim_{h \to 0}\frac{\int_{a}^b (f + h \frac{\partial f}{\partial t})  \, dx - \int_{a}^b f \, dx}{h} \\ 
&= \lim_{h \to 0}\frac{\int_{a}^b h \frac{\partial f}{\partial t} dx}{h} \\
&= \int_{a}^b \frac{\partial f}{\partial t} \, dx \\
\end{align}
$$


Thus we have shown that, if the limits of integration are constants, we can switch the order of integration and differentiation. 

But because our quench for knowledge is insatiable, let's consider the more general case as well: when the limits are not bounded by constant, but rather functions. Specifically, the case we will consider looks as follows.


$$
\frac{d}{dt} \int_{a(t)}^{b(t)} f(x, t) \, dx
$$


In this case, we see that $a$ and $b$ are each functions of variable $t$. With some thinking, it is not difficult to convince ourselves that this will indeed introduce some complications that require modifications to our original analysis. Now, not only are we slightly moving the graph of $f$ in the $t$ axis, we are also shifting the limits of integration such that there is a horizontal shift of the area box in the $x$ axis. But fear not, let's apply the same approach to answer this question.


$$
\frac{d}{dt} \int_{a(t)}^{b(t)} f(x, t) \, dx \\ 
= \lim_{h \to 0} \left[\frac{\int_{a(t + h)}^{b(t + h)} f(x, t + h) \, dx - \int_{a(t)}^{b(t)} f(x, t) \, dx}{h} \right] \\
= \lim_{h \to 0} \left[\frac{\int_{a + h\frac{da}{dt}}^{b + h\frac{db}{dt}} (f + h \frac{\partial f}{\partial t}) \, dx - \int_{a}^{b} f(x, t) \, dx}{h} \right] \\
= \lim_{h \to 0} \left[\frac{\int_{a}^{b} (f + h \frac{\partial f}{\partial t}) \, dx - \int_{a}^{a + h \frac{da}{dt}} (f + h \frac{\partial f}{\partial t}) \, dx +  \int_{b}^{b + h \frac{db}{dt}} (f + h \frac{\partial f}{\partial t}) \, dx - \int_{a(t)}^{b(t)} f(x, t) \, dx}{h} \right] \\
= \lim_{h \to 0} \left[\frac{\int_{a}^{b} h \frac{\partial f}{\partial t} \, dx - \int_{a}^{a + h \frac{da}{dt}} (f + h \frac{\partial f}{\partial t}) \, dx +  \int_{b}^{b + h \frac{db}{dt}} (f + h \frac{\partial f}{\partial t}) \, dx}{h} \right]
$$


This may appear to be a lot of computation, but all we've done is just separating out the integrals while paying attention to the domains of integration. Let's continue by doing the same for the remaining terms.


$$
\frac{d}{dt} \int_{a(t)}^{b(t)} f(x, t) \, dx \\ 
\int_{a}^{b} \frac{\partial f}{\partial t} \, dx + \lim_{h \to 0} \left[\frac{ - \int_{a}^{a + h \frac{da}{dt}} (f + h \frac{\partial f}{\partial t}) \, dx +  \int_{b}^{b + h \frac{db}{dt}} (f + h \frac{\partial f}{\partial t}) \, dx}{h} \right] \\
= \int_{a}^{b} \frac{\partial f}{\partial t} \, dx + \lim_{h \to 0} \left[ - \int_{a}^{a + h \frac{da}{dt}} \frac{\partial f}{\partial t} \, dx + \int_{b}^{b + h \frac{db}{dt}} \frac{\partial f}{\partial t} \, dx + \frac{ - \int_{a}^{a + h \frac{da}{dt}} f \, dx +  \int_{b}^{b + h \frac{db}{dt}} f \, dx}{h} \right] \\
= \int_{a}^{b} \frac{\partial f}{\partial t} \, dx + \lim_{h \to 0} \left[ \frac{ - \int_{a}^{a + h \frac{da}{dt}} f \, dx +  \int_{b}^{b + h \frac{db}{dt}} f \, dx}{h} \right]
$$


The first two terms in the limit go away since $h$ goes to zero. While the same applies to the fractional terms, one difference is that they are also divided by $h$, which is why they remain.

We have simplified quite a bit, but we still have two terms in the limit expression that we'd like to remove. We can do this by applying the definition of the integral.


$$
\int_{a}^{b} \frac{\partial f}{\partial t} \, dx + \lim_{h \to 0} \left[ \frac{ - \int_{a}^{a + h \frac{da}{dt}} f \, dx +  \int_{b}^{b + h \frac{db}{dt}} f \, dx}{h} \right] \\
= \int_{a}^{b} \frac{\partial f}{\partial t} \, dx + \lim_{h \to 0} \left[ \frac{ F(a, t) - F(a + h \frac{da}{dt}, t) + F(b + h \frac{db}{dt}, t) - F(b, t)}{h} \right] \\
= \int_{a}^{b} \frac{\partial f}{\partial t} \, dx -f(a(t), t) \frac{da}{dt} + f(b(t), t) \frac{db}{dt}
$$


And we're done! 

There are other ways of seeing the Leibniz rule, such as by interpreting it as a corollary of the Fundamental Theorem of Calculus and the Chain rule, as outlined in [here](http://www.econ.yale.edu/~pah29/409web/leibniz.pdf) (by a Professor at Yale!), but I find the geometrically motivated interpretation presented in this article to be the most intuitive. 

I hope you enjoyed reading this post. Catch you up in the next one! 