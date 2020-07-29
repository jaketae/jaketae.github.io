---
title: Revisiting Basel with Fourier
mathjax: true
toc: true
categories:
  - study
tags:
  - analysis
---

In the [last post](https://jaketae.github.io/study/zeta-prime/), we revisited the Riemann Zeta function, which we had briefly introduced in [another previous post](https://jaketae.github.io/study/basel-zeta/) on Euler's take on the famous [Basel problem](https://en.wikipedia.org/wiki/Basel_problem). It seems like math is my jam nowadays, so I decided to write another post on this topic---but this time, with some slightly different takes. 

In this post, we will explore an alternative way of solving the Basel problem using Fourier series expansion, and also discuss a alternative representations of the Basel problem in integral form. For the integral representations, I'm directly referencing [Flammable Maths](https://www.youtube.com/watch?v=yKVivOhJNRw&list=PLN2B6ZNu6xmdjskJv1udeELylfN-0Muqh&index=70), a YouTube channel that I found both entertaining and informative. 

Let's get started. 

# Basel Problem

First, let's recall what the Basel problem is. The problem is quite simple: the goal is to obtain the value of an infinite series, namely

$$
\sum_{n = 1}^\infty \frac{1}{n^2} = 1 + \frac{1}{2^2} + \frac{1}{3^2} + \cdots \tag{1}
$$

This seems like an innocuous, straightforward problem. One can easily prove, for instance, the fact that this series converges using integral approximation. However, to obtain the value of this series is a lot more difficult than it appears---it is no coincidence that this problem remained unsolved for years until Euler came along. 

While there are many ways to solve this problem---Euler's method, in particular, is one of the countless examples through which one can witness his amazing intuition and heuristic---but we will be using Fourier expansion to solve this problem, as it also provides a nice segue into the Dirichlet Eta function.

# Fourier Series

We explored the topic of Fourier expansion in [this previous post](http://jaketae.github.io/study/fourier/). To recap, from a very high level, Fourier expansion is a way of expressing some function in terms of trigonometric functions. If Taylor expansion used polynomials as the building block, Fourier expansion uses sines and cosines. 

A generic formula for the Fourier transform can be expressed as follows:

$$
f(x) 
= \frac{a_0}{2} + \sum_{n = 1}^\infty a_n \cos \left( \frac{2 \pi n}{P} x \right) + \sum_{n = 1}^\infty b_n \sin \left( \frac{2 \pi n}{P} x \right)
\tag{2}
$$


With some integration, it can be shown that

$$
a_n = \frac{2}{P} \int_{P} f(x) \cos \left( \frac{2 \pi n}{P} x \right) \, dx \tag{3}
$$

where $P$ refers to the domain of integration. For instance, if we are integrating from $- \pi$ to $\pi$, $P = 2 \pi$. A classic interval that is most commonly used is $[- \pi, \pi]$, and this is no coincidence: notice that, when $P = 2 \pi$, the Taylor series shown in (2) simplifies into the following:

$$
f(x) = \frac{a_0}{2} + \sum_{n = 1}^\infty a_n \cos(n x) + \sum_{n = 1}^\infty b_n \sin(n x)\tag{4}
$$

And indeed this is the format and the interval we will be using when constructing a Fourier series to tackle the Basel problem.

To continue, we can derive a very similar expression for $b_n$, given the specified interval from $[- \pi, \pi]$. 

$$
b_n = \frac{1}{\pi} \int_{- \pi}^{\pi} f(x) \sin(n x) \, dx \tag{5}
$$

Now that we have reviewed what Fourier series is and how we can construct it, let's jump into the Basel problem.

## Application to Basel

Just like the Taylor series, we can use Fourier expansion to represent any function continuous function. For our purposes, let's try to expand a simple polynomial function, $f(x) = x^2$, using Fourier. We can begin with $a_0$. 

$$
\begin{align}
a_0 
&= \frac{1}{\pi} \int_{-\pi}^{\pi} x^2 \, dx \\
&= \frac{1}{\pi} \cdot \frac13 x^3 \bigg\rvert_{- \pi}^{\pi} \\
&= \frac{1}{\pi} \cdot \frac23 \pi^3  \\
&= \frac23 \pi
\end{align} \tag{6}
$$

Let's continue with finding the even coefficients corresponding to the cosines. 

$$
\begin{align}
a_n 
&= \frac{1}{\pi} \int_{- \pi}^{\pi} x^2 \cos(n x) \, dx \\
&= \frac{2}{\pi} \int_{0}^{\pi} x^2 \cos(n x) \, dx 
\end{align} \tag{7}
$$

With some integration by parts, we can all agree that

$$
\frac{2}{\pi} \int_{0}^{\pi} x^2 \cos(n x) \, dx = (-1)^n \frac{4}{n^2} \tag{8}
$$

where the $(-1)^n$ terms appear because we end up plugging $\pi$ into $\cos( n x)$, a periodic function.

And we can do the same for sine. Or, even better, with the key insight that $x^2$ is an even function, we might intelligently deduce that there will be no sine terms at all, since sine functions are by nature odd. In other words, all $b_n = 0$. This can of course be shown through derivation as we have done above for the cosine coefficients. 

Therefore, putting everything together, we end up with 

$$
x^2 = \frac13 \pi^2 + \sum_{n = 1}^\infty (-1)^n \frac{4}{n^2} \cos(n x) \tag{9}
$$

If we consider the case when $x = \pi$, we have 

$$
\pi^2 = \frac13 \pi^2 + \sum_{n = 1}^\infty \frac{4}{n^2}
$$

Do you smell the basel problem in the air? The summation on the right hand side is a great sign that we are almost done in our derivation. Moving the fractional term to the left hand side, we get:

$$
\frac23 \pi^2 = \sum_{n = 1}^\infty \frac{4}{n^2}
$$

Diding both sides by 4, 

$$
\frac16 \pi^2 = \sum_{n = 1}^\infty \frac{1}{n^2} \tag{10}
$$

And there you have it, the answer to the Basel problem, solved using Fourier series!

## Dirichlet Eta Function

We can also derive a convergence value of the Dirichelt Eta function from this Fourier series as well. Recall that the Eta function looks as follows:

$$
\eta(s) = \sum_{n = 1}^\infty \frac{(-1)^n}{n^s} \tag{11}
$$

Now how can we get a Dirichelt Eta function out of the fourier series of $x^2$? Well, let's get back to (8) and think our way through. 

$$
x^2 = \frac13 \pi^2 + \sum_{n = 1}^\infty (-1)^n \frac{4}{n^2} \cos(n x) \tag{8}
$$

One noteworthy observation is that we already have $(-1)^n$ in the summation, which looks awfully similar to the Dirichlet Eta function. Since we want to get rid of the cosine term, we can simply set $x = 0$---this will make all cosine terms evaluate to 1, effectively eliminating them from the expression. Then, we get

$$
0 = \frac13 \pi^2 + \sum_{n = 1}^\infty (-1)^n \frac{4}{n^2} \tag{8}
$$

With a very small bit of algebra, we end up with 

$$
- \frac{1}{12} \pi^2 =  \sum_{n = 1}^\infty \frac{(-1)^n}{n^2}
$$

And there we have it, the value of $\eta(2)$! It's interesting to see how all this came out of the fourier series of $x^2$. 

# Integral Representations


In this section, we will be taking a look at some interesting representations of the Basel problem, mysteriously packaged in integrals. At a glance, it's somewhat unintuitive to think that an infinite summation problem can be stated as an integral in exact terms; however, the translation from summation to integrals are not out of the blue. Using things like Taylor series, it is in fact possible to show that the Basel problem can be stated as an integral. 

## Natural Logarithm

For instance, consider this integral

$$
\int_{0}^1 \frac{\ln(x)}{x - 1} \, dx \tag{9}
$$

One thing I am starting to realize these past few days is that some of these integrals are extremely difficult despite being deceptively simple in their looks. This is a good example. 

To get started, we might consider making a quick change of variables, namely $x - 1 = t$. This will effectively get rid of the rather messy-looking denominator sitting in the fraction. 

$$
\int_{-1}^0 \frac{\ln(1 + t)}{t} \, dt \tag{10}
$$

To make further progress, at this point let's consider the Taylor series expansion of $\ln(1 + t)$. We can derive this by considering the following integral:

$$
\int \frac{1}{1 + t} \, dt \tag{11}
$$

since this integral evaluates to $\ln(1 + t)$. 

One way to look at (10) would be to consider it as a sum of some geometric series whose first term begins with 1 and has a constant ratio of $-t$. In other words, 

$$
\begin{align}
\int \frac{1}{1 + t} \, dt
&= \int 1 - t + t^2 - t^3 + \cdots  \, dt \\
&= \int \sum_{n = 0}^\infty (- t)^n \, dt
\end{align} \tag{12}
$$

Here is where a bit of complication comes in. Turns out that under certain conditions, we can exchange the summation and the integral (or, more strictly speaking, the limit and the integral), using things like the dominating convergence theorem of Fubini's theorem. However, these are topics for another post. For now, we will assume that this trick is legal and continue on. Now we have

$$
\sum_{n = 0}^\infty \int (- t)^n \, dt \\
= \sum_{n = 0}^\infty (-1)^n \frac{t^{n + 1}}{n + 1} \tag{13}
$$

Now that we have a summation representation of $\ln$, let's move onto (11). 

$$
\int_{-1}^0 \sum_{n = 0}^\infty (-1)^n \frac{t^n}{n + 1} \, dt
$$

We use the same trick we used earlier to interchange the summation and the integral. This gives us 

$$
\sum_{n = 0}^\infty \int_{-1}^0 (-1)^n \frac{t^n}{n + 1} \, dt \\
= \sum_{n = 0}^\infty (-1)^n \frac{(-1)^n}{(n + 1)^2} \, dt \tag{14}
$$

Since we have to terms with negative ones with the same exponent, we can safely remove both of them:

$$
\sum_{n = 0}^\infty \frac{1}{(n + 1)^2} \, dt \tag{15}
$$

And notice that we now have the Basel problem! If you plug in $n=0$ and increment $n$ from there, it is immediately apparent that this is the case. So there we have it, the integral representation of the Basel problem!

## Double Integral

Let's look at another example, this time using a double integral representation. 

The motivation behind this approach is simple. 

$$
\begin{align}
\int_{0}^1 x^n \, dx 
&= \frac{1}{n + 1} x^{n + 1} \bigg \rvert_{0}^{1} \\
&= \frac{1}{n + 1}
\end{align} \tag{16}
$$

This is a useful result, since it means that we can express the Basel problem as an integral of two different variables. 

$$
\int_{0}^1 \int_{0}^1 x^n  t^n \, dx \, dt = \frac{1}{(n + 1)^2} \tag{17}
$$

Now, all we need is a summation expression before the integration. 

$$
\sum_{n = 0}^\infty \int_{0}^1 \int_{0}^1 x^n  t^n \, dx \, dt = \sum_{n = 0}^\infty \frac{1}{(n + 1)^2} \tag{18}
$$

And now we are basically back to the Basel problem. 

Note that we can also use the interchange of integral and summation technique again to reexpress (19) as shown below.

$$
\int_{0}^1 \int_{0}^1 \sum_{n = 0}^\infty (x t)^n \, dx \, dt \tag{19}
$$

Notice that now we have a geometric series, which means that now we can also express this integral as 

$$
\int_{0}^1 \int_{0}^1 \frac{1}{1 - x t} \, dx \, dt \tag{20}
$$

Like this, there are countless ways of using integrals to express the Basel problem. This representation, in particular, could be understood as an integral over a unit square in the cartesian coordinate over a bivariate function, $f(x, t)$. 

# Conclusion

In this post, we took a look at a new way of approaching the Basel problem using Fourier expansion. We also looked at some interesting integral representations of the Basel problem. 

While a lot of this is just simple calculus and algebra, I nonetheless find it fascinating how the Basel problem can be approached from so many different angles---hence my renewed respect for Euler and other mathematicians who wrestled with this problem hundreds of years ago. As simple as it appears, there are so many different techniques and modes of analysis we can use to approach the problem. It was nice exercise and review of some calculus techniques. 

I've been digging more into the curious interchange of integral and summation recently, and when this operation is allowed, if at all. Turns out that this problem is slightly more complicated than it appears and requires some understanding of measure theory, which I had tried getting into a few months ago without much fruition. Hopefully this time, I'll be able to figure something out, or at the very least gain some intuition on this operation. 

I hope you've enjoyed reading this post. Catch you up in the next one.
