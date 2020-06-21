---
title: "Basel, Zeta, and some more Euler"
date: 2019-11-26
categories:
  - study
tags:
  - euler
  - analysis
---

The more I continue my journey down the rabbit hole of mathematics, the more often I stumble across one name: Leonhard Euler. Nearly every concept that I learn, in one way or another, seems to be built on top of some strand of his work, not to mention the unending list of constants, formulas, and series that bears his name. It is simply mind-blowing to imagine that a single person could be so creative, inventive, and productive to the extent that the field of mathematics would not be where it is today had it not been for his birth on April 15, 1707.  

# The Basel Problem

Why such intensive fanboying, you might ask. Well, let’s remind ourselves of the fact that the interpolation of the factorial through the Gamma function was spearheaded by Euler himself. But this is just the start of the beginning. Consider, for example, the [Basel problem], an infamous problem that mathematicians have been trying to solve for nearly a century with no avail, until the 28-year-old Euler came to the rescue. The Basel problem can be stated as follows:

<script type="text/javascript" async  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

$$\frac{1}{1^2} + \frac{1}{2^2} + \frac{1}{3^2} + \dots = \sum_{n = 1}^\infty \frac{1}{n^2} = ? $$

At a glance, this seems like a fairly simple problem. Indeed, we know that this series converges to a real value. We also know that integration would give us a rough approximation. However, how can evaluate this series with exactitude? 

[Euler’s solution], simple and elegant, demonstrates his genius and acute heuristics. Euler begins his exposition by analyzing the Taylor expansion of the sine function, which goes as follows:

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} \dots = \sum_{n = 0}^\infty (-1)^n \frac{x^{2n + 1}}{(2n + 1)!}$$

Dividing both sides by $$x$$, we obtain the following:

$$\frac{\sin(x)}{x} = 1 - \frac{x^2}{3!} + \frac{x^4}{5!} \dots = \sum_{n = 0}^\infty (-1)^n \frac{x^{2n}}{(2n + 1)!}$$

Now it’s time to follow Euler’s amazing intuition. If we take a close look at the equation $$\frac{\sin(x)}{x} = 0$$, we can convince ourselves that its solutions will adhere to the form $$n\pi$$, where $$n$$ is a non-zero integer between $$[- \infty, \infty]$$. This is expected given the periodic behavior of the sine function and its intercepts with the $$x$$-axis. Given these zeros, we can then reconstruct the original function $$\frac{\sin(x)}{x}$$ as an infinite product using the [fundamental theorem of algebra], or more specifically, [Weierstrass factorization]. 

$$\frac{\sin(x)}{x} = (1 - \frac{x}{\pi})(1 + \frac{x}{\pi})(1 - \frac{x}{2 \pi})(1 + \frac{x}{2 \pi}) \dots$$

Let’s try to factor out the coefficient of the $$x^2$$ term through some elementary induction. First, we observe that calculating the product of the first two terms produces the following expression:

$$
\frac{\sin(x)}{x} = (1 - \frac{x^2}{\pi^2})(1 - \frac{x}{2 \pi})(1 + \frac{x}{2 \pi}) \dots
$$


Then, we can express the target coefficient, denoted by $$C$$, as follows:

$$C = C’ - \frac{1}{\pi^2}$$

Where $$C’$$ denotes the coefficient of $$x^2$$ obtained by expanding the rest of the terms following $$(1 - \frac{x^2}{\pi^2})$$ in the infinite product above. If we repeat this process once more, a clear pattern emerges:

$$C = C’ - \frac{1}{\pi^2} = C’’ - \frac{1}{\pi^2} - \frac{1}{4 \pi^2}$$

Iterating over this process will eventually allow us to express our target coefficient as a sum of inverse squares multiplied by some constant, in this case $$\pi^2$$:

$$C = - (\frac{1}{\pi^2} + \frac{1}{4 \pi^2} + \frac{1}{9 \pi^2} \dots) = - \frac{1}{\pi^2} \sum_{n = 1}^\infty \frac{1}{n^2}$$

But then we already know the value of $$C$$ from the modification of the Taylor polynomial for sine we saw earlier, which is $$- \frac{1}{6}$$! Therefore, the Basel problem reduces to the following:

$$- \frac{1}{6} = - \frac{1}{\pi^2} \sum_{n = 1}^\infty \frac{1}{n^2}$$

Therefore,

$$\sum_{n = 1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$$

And there we have the solution to the Basel problem. In hindsight, solving the Basel problem is not rocket science; it is a mere application of the Taylor polynomial, coupled with some modifications and operations to mold the problem into a specific form. However, it takes extreme clairvoyance to see the link between the Basel problem and the Taylor polynomial of the sine function. At this point, it doesn’t even surprise us to know that Euler applied this line of thinking to calculate the value of the sum of higher order inverses.

An interesting corollary of Euler’s solution to the Basel problem is the [Wallis product], which is a representation of the quantity $$\frac{\pi}{2}$$ as an infinite product, as shown below:

$$\frac{\pi}{2} = (\frac{2}{1} \cdot \frac{2}{3}) \cdot (\frac{4}{3} \cdot \frac{4}{5}) \cdot (\frac{6}{5} \cdot \frac{6}{7}) \dots$$

It seems mathematically unintuitive to say that an irrational number such as $$\pi$$ can be expressed as a product of fractions, which is a way of representing rational numbers. However, we can verify the soundness of the Wallis product by substituting $$\frac{\pi}{2}$$ for $$x$$ in (1):

$$\frac{2}{\pi} = \prod_{n = 1}^\infty (1 - \frac{1}{4n^2})$$

Taking the reciprocal of this entire expression reveals the Wallis product.

$$\frac{\pi}{2} = \prod_{n = 1}^\infty \frac{4 n^2}{(4 n^2 - 1)} = \prod_{n = 1}^\infty \frac{2n}{(2n - 1)} \cdot \frac{2n}{(2n + 1)} = \frac{2}{1} \cdot \frac{2}{3} \cdot \frac{4}{3} \dots$$

# The Zeta Function

The Basel problem is a specific case of the [Riemann zeta function], whose general form can be written as follows. 

$$\zeta(s) = \sum_{n = 1}^\infty \frac{1}{n^s}$$

A small digression: when $$s = 3$$, the zeta function converges to a value known as [Apéry’s constant], eponymously named after the French mathematician who proved its irrationality in the late 20th century. Beyond the field of analytics and pure math, the zeta function is widely applied in fields such as physics and statistics. Perhaps we will explore these topics in the future. 

So back to the zeta function. In the later segment of his life, Euler found a way to express the zeta function as, you guessed it, an infinite product. This time, however, Euler did not rely on Taylor polynomials. Instead, he employed a more general approach to the problem. It is here that we witness Euler’s clever manipulation of equations again. 

We commence from the zeta function, whose terms are enumerated below. 

$$\zeta(s) = 1 + \frac{1}{2^s} + \frac{1}{3^s} + \frac{1}{4^s} + \dots $$

Much like how we multiply the ratio to a geometric sequence to calculate its sum, we adopt a similar approach by multiplying the second term, $$\frac{1}{2^s}$$ to the entire expression. This operations yields

$$\frac{1}{2^s} \zeta(s) = \frac{1}{2^s} + \frac{1}{4^s} + \frac{1}{6^s} + \dots $$

By subtracting this modified zeta function from the original, we derive the following expression below.

$$(1 - \frac{1}{2^s}) \zeta(s) = 1 + \frac{1}{3^s} + \frac{1}{5^s} + \dots$$

Now, we only have what might be considered as the odd terms of the original zeta function. We then essentially repeat the operation we have performed so far, by multiplying the expression by $$\frac{1}{3^s}$$ and subtracting the result from the odd-term zeta function. 

$$\frac{1}{3^s}(1 - \frac{1}{2^s}) \zeta(s) = \frac{1}{3^s} + \frac{1}{9^s} + \frac{1}{15^s} + \dots$$

$$(1 - \frac{1}{2^s})(1 - \frac{1}{3^s}) \zeta(s) = 1 + \frac{1}{5^s} + \frac{1}{7^s} + \dots$$

It is not difficult to see that iterating through this process will eventually yield Euler’s product identity for the zeta function. The key to understanding this identity is that only prime numbers will appear as a component of the product identity. We can see this by reminding ourselves of the clockwork behind the [sieve of Eratosthenes], which is basically how the elimination and factorization works in the derivation of Euler’s identity. Taking this into account, we can deduce that Euler’s identity will take the following form:

$$\zeta(s) = (1 - \frac{1}{2^s})(1 - \frac{1}{3^s})(1 - \frac{1}{5^s})(1 - \frac{1}{7^s}) \dots = \prod_{p \in prime} (1 - p^{-s})$$

This expression is Euler’s infinite product representation of the zeta function. 

# Conclusion

These days, I cannot help but fall in love with Euler’s works. His proofs and discoveries are simple and elegant yet also fundamental and deeply profound, revealing hidden relationships between numbers and theories that were unthought of during his time. I tend to avoid questions like “who was the best $$X$$ in history” because they most often lead to unproductive discussions that obscure individual achievements amidst meaningless comparisons, but I dare profess here my belief that only a handful of mathematicians can rival Euler in terms of his genius and prolific nature. 

That is enough Euler for today. I’m pretty sure that this is not going to be the last post on Euler given the sheer amount of work he produced during his lifetime. My exploration of the field of mathematics is somewhat like a random walk, moving from one point to another with no apparent pattern or purpose other than my interest and Google’s search suggestions, but my encounter with Euler will recur continuously throughout this journey for sure. But for now, I’m going to take a brief break from Euler and return back to the topic of good old statistical analysis, specifically [Bayesian methods] and Monte Carlo methods. Catch you up in the next one!




[Basel problem]: https://en.wikipedia.org/wiki/Basel_problem

[Euler’s solution]: http://www.17centurymaths.com/contents/euler/e041tr.pdf

[Weierstrass factorization]: https://en.wikipedia.org/wiki/Weierstrass_factorization_theorem

[fundamental theorem of algebra]: https://en.wikipedia.org/wiki/Fundamental_theorem_of_algebra

[Riemann zeta function]: https://en.wikipedia.org/wiki/Riemann_zeta_function#Euler_product_formula

[Apéry’s constant]: https://en.wikipedia.org/wiki/Apéry%27s_constant

[sieve of Eratosthenes]: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes

[Wallis product]: https://en.wikipedia.org/wiki/Wallis_product

[Bayesian methods]: https://jaketae.github.io/study/bayes/


