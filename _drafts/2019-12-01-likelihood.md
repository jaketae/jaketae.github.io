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





[earlier post]: https://jaketae.github.io/study/bayes/
[likelihood function]: https://en.wikipedia.org/wiki/Likelihood_function
[Z-score]: https://en.wikipedia.org/wiki/Standard_score














