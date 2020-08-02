---
title: Gamma and Zeta
mathjax: true
toc: true
categories:
  - study
tags:
  - analysis
---

Maintaining momentum in writing and self-learning has admittedly been difficult these past few weeks since I’ve started my internship. Normally, I would write one post approximately every four days, but this routine is no longer the norm. To my defense, I’ve been learning a ton about Django and backend operations like querying and routing, and I might write a post about these in the future. 

But for today, I decided to revisit a topic we’ve previously explored on this blog, partially in the hopes of using nostalgia as positive energy in restarting my internal momentum. I must also note that I meant to write this post for a very long time after watching [this video](https://www.youtube.com/watch?v=ctG4YgMs74w) by [blackpenredpen](https://www.youtube.com/channel/UC_SvYP0k05UKiJ_2ndB02IA) whose videos have been a source of mathematical learning and inspiration for me. 

Let’s talk about the Gamma and Zeta functions.

# The Greek Letters

Before we begin the derivation, perhaps it's a good idea to review what the two greek letter functions are.

## Gamma (and Pi)

The Gamma function is written as 


$$
\Gamma(x) = \int_0^\infty t^{x - 1} e^{- t} \, dt \tag{1}
$$



And we all know that the Gamma function can be seen as an interpolation of the factorial function, since for non-negative integers, the following relationship stands:


$$
\Gamma(x) = (x - 1)! \tag{2}
$$


Note that there is also a variant of the Gamma function, known as the Pi function, which has somewhat of a nicer form:


$$
\Pi(x) = \int_0^\infty t^x e^t \, dt = x! \tag{3}
$$


To the mathematically uninformed self, the Pi function seems a lot more tractable and intuitive. Nonetheless, the prevailing function is Euler's Gamma function instead of Gauss's Pi function. The reasons for Gamma's dominance over Pi is discussed extensively in this [math overflow thread](https://mathoverflow.net/questions/20960/why-is-the-gamma-function-shifted-from-the-factorial-by-1). At any rate, it's both interesting and yet also unsurprising to see that these two functions are brainchildren of Gauss and Euler, two names that arguably appear the most in the world of math.

## Riemann Zeta Function

The Riemann zeta function is perhaps one of the most famous functions in the world of analysis. It is also sometimes referred to as the Euler-Riemann zeta function, but at this point, prefixing something with "Euler" loses significance since just about everything in mathematics seems to have some Euler prefix in front of it. 

The Riemann zeta function takes the following form:


$$
\zeta(s) = \sum_{n = 1}^\infty \frac{1}{n^s} \tag{4}
$$


But this definition, as simple and intuitive as it is, seems to erroneously suggest that the Riemann zeta function is only defined over non-negative integers. This is certainly not the case. In fact, the reason why the Riemann zeta function is so widely studied in mathematics is that its domain ranges over the complex number plane. While we won't be discussing the complexity of the Riemann zeta function in this regard (no pun intended), it is nonetheless important to consider about how we might calculate, say, $\zeta(1.5)$. This is where the Gamma function comes in.

# Derivation

As hinted earlier, an alternative definition of the Riemann zeta function can be constructed using the Gamma function that takes the Riemann zeta beyond the obvious realm of integers and into the real domain. 

We first start with a simple change of variables. Specifically, we can substitute $t$ for $nu$. This means that $dt = n \,du$, using which we can establish the following:


$$
\begin{align}
\Gamma(x) 
&= \int_0^\infty t^{x - 1} e^{- t} \, dt \\
&= \int_0^\infty (nu)^{x - 1} e^{- nu} n \, du \\
\end{align}
\tag{5}
$$


With some algebraic implications, we end up with


$$
\begin{align}
\Gamma(x) 
&= \int_0^\infty n^x u^{x - 1} e^{- nu} \, du \\
&= n^x \int_0^\infty  u^{x - 1} e^{- nu} \, du \\
\end{align}
\tag{6}
$$


Dividing both sides by $n^x$, we get


$$
\frac{1}{n^x} \Gamma(x) = \int_0^\infty  u^{x - 1} e^{- nu} \, du \tag{7}
$$


All the magic happens when we cast a summation on the entire expression:


$$
\begin{align}
\Gamma(x) \sum_{n = 1}^\infty \frac{1}{n^x} 
&= \Gamma(x) \zeta(x) \\
&= \sum_{n = 1}^\infty \int_0^\infty  u^{x - 1} e^{- nu} \, du 
\end{align}
\tag{8}
$$


Notice that now we have the Riemann zeta function on the left hand side. All we have to do is to clean up what is on the right. As it stands, the integral is not particularly tractable; however, we can swap the integral and the summation expression to make progress. I still haven't figured out the details of when this swapping is possible, which has to do with absolute divergence, but I will be blogging about it in the future once I have a solid grasp of it, as promised before. 


$$
\begin{align}
\Gamma(x) \zeta(x) 
&= \sum_{n = 1}^\infty \int_0^\infty  u^{x - 1} e^{- nu} \, du \\
&= \int_0^\infty  u^{x - 1} \left( \sum_{n = 1}^\infty e^{- nu} \right) \, du 
\end{align}
\tag{9}
$$


The expression in the parentheses is just a simple sum of geometric series, which we know how to calculate. Therefore, we obtain


$$
\int_0^\infty  u^{x - 1} \left( \frac{e^{- u}}{1 - e^{-u}} \right) \, du \tag{10}
$$


To make this integral look more nicer into a form known as the Bose integral, let's multiply both the numerator and the denominator by $e^{- u}$. After some cosmetic simplifications, we end up with


$$
\int_0^\infty \frac{u^{x - 1}}{e^{-u} - 1}  \, du \tag{11}
$$


Putting everything together, now we have derived a nice expression that places both the Riemann zeta and the Gamma functions together:


$$
\Gamma(x) \zeta(x) = \int_0^\infty \frac{u^{x - 1}}{e^{-u} - 1}  \, du \tag{12}
$$


Or, alternatively, a definition of the Riemann zeta in terms of the Gamma:


$$
\zeta(x) = \frac{1}{\Gamma(x)} \int_0^\infty \frac{u^{x - 1}}{e^{-u} - 1}  \, du \tag{13}
$$


And indeed, with (13), we can evaluate the Riemann zeta function at non-integer points as well. This is also the definition of the Riemann zeta function introduced in [Wikipedia](https://en.wikipedia.org/wiki/Riemann_zeta_function#Definition). The article also notes, however, that this definition only applies in a limited number of cases. This is because we've assumed, in using the summation of the geometric series formula, the fact that $\lvert e^{-u} \rvert < 1$.   

# Conclusion

Today's post was a short yet very interesting piece on the relationship between the Gamma and the Riemann zeta. One thing I think could have been executed better is the depth of the article---for instance, what is the Bose integral and when is it used? I've read a few comments on the original YouTube video by blackpenredpen, where people were saying that the Bose integral is used in statistical mechanics and the study of black matter, but discussing that would require so much domain knowledge to cover. Regardless, I think the theoretical aspect of this derivation is interesting nonetheless. One thing I must do is writing a post on divergence and when the interchange of summation and integrals can be performed. 

I was originally planning to write a much longer article dividing deep into the Gamma and the Beta function as well as their distributions. However, I realized that what I need at this point in time is producing output and reorienting myself back to self-studying blogger mode, perhaps taking a brief hiatus from the grinding intern spending endless hours in Sublime text with Django (of course, I'm doing that because I enjoy and love the dev work). In the end, we all need a healthy balance between many things in life, and self-studying and working are definitely up on that list for me. Hopefully I can find the middle ground that suits me best. 

I hope you've enjoyed reading this post. Catch you up in the next one!