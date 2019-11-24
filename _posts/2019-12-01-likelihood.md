---
title: (Maximum) Liklihood and Probability
last_modified_at: 2019-12-01 9:50:00 +0000
categories:
  - study
tags:
  - math
---

"I think that's very unlikely."

"No, you're probably right."

These are just some of the many remarks we use in every day conversations to express our beliefs. Linguistically, words such as "probably" or "likely" serve to qualify the strength of our professed belief, that is, we express a degree of uncertainty involved with a given statement. 

In today's post, I suggest that we scrutinize the concept of likelihood---what it is, how we calculate it, and most importantly, how different it is from probability. Although the vast majority of us tend to conflate likelihood and probability in daily conversations, mathematically speaking, these two are distinct concepts, though closely related. After concretizing this difference, we then move onto a discussion of maximum likelihood, which is a useful tool frequently employed in Bayesian statistics. Without further ado, let's jump right in.

# Likelihood vs Probability

As we have seen in an [earlier post] on Bayesian analysis, likelihood tells us---and pardon the circular definition here---how likely a certain parameter is given some data. In other words, the [likelihood function] answers the question: provided some list of observed or sampled data $$D$$, what is the likelihood that our parameter of interest takes on a certain value $$\theta$$? In mathematical notation, this idea might be transcribed as:

$$L(\theta \mid D)$$

Notice that likelihood flips the question that we ask when analyzing probability. Probability primarily deals with the question of: given some parameter $$\theta$$ for a distribution, what is the probability that we observe data $$D$$? Hence the statement that, the farther away a data point is from a normal distribution, *i.e.* the higher its [Z-score], the less probability there is that we would observe such a sample. Likelihood and probability, therefore, seem to ask similar questions, but in fact they approach the same phenomenon from opposite angles, one with a focus on the paramter and the other on data. 

Let's develop more intuition by analyzing the difference between likelihood and probability from a graphical standpoint. To get started, recall the that

$$P(a < X < b \mid \theta) = \int_a^b f(p) dp$$

This is the good old definition of probability as defined for a continuous random varriable $$X$$, given some probability density function $$f(p)$$ with parameter $$\theta$$. Graphically speaking, we can consider probability as the area or volume under the probability density function, which may be a curve, plane, or a hyperplane depending on the dimensionality of our context. 

<figure>
	<img src="/assets/images/probability.png">
	<figcaption>Figure 1: Representation of probability as area</figcaption>
</figure>

Unlike probability, likelihood is best understood as a point estimate on the PDF. Imagine having two different distributions with different parameters. Likelihood is an estimate we can use to see which of these two distributions better explain the data we have in our hands. Intuitively, the closer the mean of the distribution is to the obseved data point, the more likely the parameters for the distribution would be. We can see this in action with a simple line of code.

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

This code block creates two distributions of different parameters, $$N_1~(1, 0)$$ and $$N_2~(0.5, 0.7)$$. Then, we assume that a sample of value 1 is observed. Then, we can compare the likelihood of the two parameters given this data by compairng the probability density of the data for each of the two distributions. 

<figure>
	<img src="/assets/images/likelihood.png">
	<figcaption>Figure 1: Representation of probability as area</figcaption>
</figure>

In this case, $$N_2$$ seems more likely, *i.e.* it better explains the data $$X = 1$$ since $$L(\theta_{N_1} \mid 1) \approx 0.4416$$, which is larger than $$L(\theta_{N_2} \mid 1) \approx 0.2420$$. 

To sum up, likelihood is something that we can say about a distribution, specifically the parameter of the distribution. On the other hand, probabilities are quantities that we ascribe to individual data. Although these two concepts are easy to conflate, and indeed there exists an important relationship between them explained by Bayes' theorem, yet they should not be conflated in the world of mathematics. At the end of the day, both of them provide interesting ways to analyze statistical phenomena. 

# Maximum Likelihood

Maximum likelihood is a concept that is very important for Bayesian statistics, as we will see in later posts as we explore MCMC algorithms, most notably Metropolis-Hastings. 



[earlier post]: https://jaketae.github.io/study/bayes/
[likelihood function]: https://en.wikipedia.org/wiki/Likelihood_function
[Z-score]: https://en.wikipedia.org/wiki/Standard_score














