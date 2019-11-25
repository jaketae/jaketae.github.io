---
title: Likelihood and Probability
last_modified_at: 2019-12-01 9:50:00 +0000
categories:
  - study
tags:
  - math
---

"I think that's very unlikely." "No, you're probably right."

These are just some of the many remarks we use in every day conversations to express our beliefs. Linguistically, words such as "probably" or "likely" serve to qualify the strength of our professed belief, that is, we express a degree of uncertainty involved with a given statement. 

In today's post, I suggest that we scrutinize the concept of likelihood---what it is, how we calculate it, and most importantly, how different it is from probability. Although the vast majority of us tend to conflate likelihood and probability in daily conversations, mathematically speaking, these two are distinct concepts, though closely related. After concretizing this difference, we then move onto a discussion of maximum likelihood, which is a useful tool frequently employed in Bayesian statistics. Without further ado, let's jump right in.

# Likelihood vs Probability

As we have seen in an [earlier post] on Bayesian analysis, likelihood tells us---and pardon the circular definition here---how likely a certain parameter is given some data. In other words, the [likelihood function] answers the question: provided some list of observed or sampled data $$D$$, what is the likelihood that our parameter of interest takes on a certain value $$\theta$$? One measurement we can use to answer this question is simply the probability density of the observed value of the random variable at that distribution. In mathematical notation, this idea might be transcribed as:

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

$$L(\theta \mid D) = P(D \mid \theta) \tag{1}$$

At a glance, likelihood seems to equal probability---after all, that is what the equation (1) seems to suggest. But first, let's clarify the fact that $$P(D \mid \theta)$$ is probability density, not probability. Moreover, the interpretation of probability density in the context of likelihood is different from that which arises when we discuss probability; likelihood attempts to explain the fit of observed data by altering the distribution parameter. Probability, in contrast, primarily deals with the question of how probable the observed data is given some parameter $$\theta$$. Likelihood and probability, therefore, seem to ask similar questions, but in fact they approach the same phenomenon from opposite angles, one with a focus on the parameter and the other on data. 

Let's develop more intuition by analyzing the difference between likelihood and probability from a graphical standpoint. To get started, recall the that

$$P(a < X < b \mid \theta) = \int_a^b f(p) dp$$

This is the good old definition of probability as defined for a continuous random varriable $$X$$, given some probability density function $$f(p)$$ with parameter $$\theta$$. Graphically speaking, we can consider probability as the area or volume under the probability density function, which may be a curve, plane, or a hyperplane depending on the dimensionality of our context. 

<figure>
	<img src="/assets/images/probability.png">
	<figcaption>Figure 1: Representation of probability as area</figcaption>
</figure>

Unlike probability, likelihood is best understood as a point estimate on the PDF. Imagine having two disparate distributions with distinct parameters. Likelihood is an estimate we can use to see which of these two distributions better explain the data we have in our hands. Intuitively, the closer the mean of the distribution is to the observed data point, the more likely the parameters for the distribution would be. We can see this in action with a simple line of code.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal(x, option=True):
    if option:
        mu, sigma = 0, 1
    else:
        mu, sigma = 0.5, 0.7
    return norm.pdf(x, mu,sigma)

x = np.linspace(-5, 5, 100)
p_x = [1, 1]
p_y = [normal(1), normal(1, option=False)]

plt.style.use("seaborn")
plt.plot(x, normal(x), color="skyblue")
plt.plot(x, normal(x, option=False), color="violet")
plt.plot(1, 0, marker='o', color="black")
plt.vlines(p_x, 0, p_y, linestyle="--", color="black", alpha=0.5, linewidth=1.2)
for i, j in zip(p_x, p_y):
    plt.text(i, j, 'L = {:.4f}'.format(j))
plt.title("Likelihood as Height")
plt.xlabel("X")
plt.ylabel("Density")
plt.show()
```

This code block creates two distributions of different parameters, $$N_1~(1, 0)$$ and $$N_2~(0.5, 0.7)$$. Then, we assume that a sample of value 1 is observed. Then, we can compare the likelihood of the two parameters given this data by comparing the probability density of the data for each of the two distributions. 

<figure>
	<img src="/assets/images/likelihood.png">
	<figcaption>Figure 2: Representation of likelihood as height</figcaption>
</figure>

In this case, $$N_2$$ seems more likely, *i.e.* it better explains the data $$X = 1$$ since $$L(\theta_{N_1} \mid 1) \approx 0.4416$$, which is larger than $$L(\theta_{N_2} \mid 1) \approx 0.2420$$. 

To sum up, likelihood is something that we can say about a distribution, specifically the parameter of the distribution. On the other hand, probabilities are quantities that we ascribe to individual data. Although these two concepts are easy to conflate, and indeed there exists an important relationship between them explained by Bayes' theorem, yet they should not be conflated in the world of mathematics. At the end of the day, both of them provide interesting ways to analyze the organic relationship between data and distributions. 

# Maximum Likelihood 

[Maximum likelihood estimation], or MLE in short, is an important technique used in many subfields of statistics, most notably [Bayesian statistics]. As the name suggests, the goal of maximum likelihood estimation is to find the parameters of a distribution that maximizes the probability of observing some given data $$D$$. In other words, we want to find the optimal way to fit a distribution to the data. As our intuition suggests, MLE quickly reduces into an optimization problem, the solution of which can be obtained through various means, such as Newton's method or gradient descent. For the purposes of this post, we look at ways to approach MLE problems using the former. 

The best way to demonstrate how MLE works is through examples. In this post, we look at simple examples of maximum likelihood estimation in the context of normal and exponential distributions. 

## Normal Distribution

We have never formally discussed normal distributions on this blog yet, but it is such a widely used, commonly referenced distribution that I decided to jump into MLE with this example. But don't worry---we will derive the normal distribution in a future post, so if any of this seems overwhelming, you can always come back to this post for reference. 

The probability density function for the normal distribution, with parameters $$\mu$$ and $$\sigma$$, can be written as follows:

$$P(X = x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{-(x - \mu)^2}{2 \sigma^2}}$$

Assume we have a list of observations that correspond to the random variable of interest, $$X$$. For each $$x_i$$ in the sample data, we can calculate the likelihood of a distribution with parameters $$\theta = (\mu, \sigma)$$ by calculating the probability densities at each point of the PDF where $$X = x_i$$. We can then make the following statement about these probabilities:

$$L(\theta \mid x_1, x_2, \dots x_n) = P(X = x_1, x_2, \dots x_n \mid \theta) = \prod_{i = 1}^n P(X = x_i \mid \theta)$$

In other words, to maximize the likelihood simply means to find the value of a parameter that which maximizes the product of probabilities of observing each data point. The assumption of independence allows us to use multiplication to calculate the likelihood in this manner. Applied in the context of normal distributions with $$n$$ observations, the likelihood function can therefore be calculated as follows:

$$L = \prod_{i = 1}^n \frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{-(x_i - \mu)^2}{2 \sigma^2}} \tag{2}$$

But finding the maximum of this function can quickly turn into a nightmare. Recall that we are dealing with distributions here, whose PDFs are not always the simplest and the most elegant-looking. If we multiply $$n$$ terms of the normal PDF, for instance, we would end up with a giant exponential term. To prevent this fiasco, we can introduce a simple transformation: logarithms. Log is a [monotonically increasing function], which is why maximizing some function $$f$$ is equivalent to maximizing the log of that function, $$\log(f)$$. Moreover, the log transformation expedites calculation since logarithms restructure multiplication as sums. 

$$\log(ab) = \log(a) + \log(b)$$

With that in mind, we can construct a log equation for MLE from (2) as shown below. Because we are dealing with Euler’s number, $$e$$, the natural log is our preferred base.

$$\ln(L) = 


And here is a gentle reminder that the goal of maximum likelihood estimation is to find the parameter of a distribution that best explains given data. To proceed further, let's consider a sample of integers that will serve as our observed data. 

```python
numbers_list = [4, 5, 7, 8, 8, 9, 10, 5, 2, 3, 5, 4, 8, 9]
```

(This article is currently in progress.)



[link 1]: https://medium.com/@rrfd/what-is-maximum-likelihood-estimation-examples-in-python-791153818030
[link 2]: https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
[link 3]: https://web.sonoma.edu/users/w/wilsonst/papers/Normal/default.html
[normal]: https://www.youtube.com/watch?v=Dn6b9fCIUpM&t=241s
[exponential]: https://www.youtube.com/watch?v=p3T-_LMrvBc&t=479s]
[binomial]: https://www.youtube.com/watch?v=4KKV9yZCoM4&t=330s

[earlier post]: https://jaketae.github.io/study/bayes/
[likelihood function]: https://en.wikipedia.org/wiki/Likelihood_function
[Maximum likelihood estimation]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
[Bayesian statistics]: https://en.wikipedia.org/wiki/Bayesian_statistics
[monotonically increasing function]: https://en.wikipedia.org/wiki/Monotonic_function
