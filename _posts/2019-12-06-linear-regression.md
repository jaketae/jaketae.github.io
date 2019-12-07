---
title: Linear Regression, in Two Ways
date: 2019-12-06
mathjax: true
categories:
  - study
tags:
  - math
  - python
---

If there is one thing I recall most succinctly from my high school chemistry class, it is how to use Excel to draw basic plots. In the eyes of a naive freshman, visualizations seemed to add an air of professionalism. So I would always include a graph of some sort in my lab report, even when I knew they were superfluous. The final icing on the cake? A fancy regression with some r-squared. 

In today’s post, I want to revisit what used to be my favorite tinkering toy in Excel: regression. More specifically, we’ll take a look at linear regression, which deals with straight lines and planes instead of curved surfaces. Although it sounds simple, the linear regression model is still widely used because it not only provides a clearer picture of obtained data, but can also be used to make predictions based on previous observations. Linear regression is also incredibly simple to implement using existing libraries in programming languages such as Python, as we will later see in today’s post. That was a long prologue—--let’s jump right in.

# Back to Linear Algebra

## Problem Setup

In this section, we will use linear algebra to understand regression. An important theme in linear algebra is orthogonality. How do we determine if two vectors---or more generally, two subspaces---are orthogonal to each other? How do we make two non-orthogonal vectors orthogonal? (Hence Gram-Schmidt.) In our case, we love orthogonality because they are key to deriving the equation for the line of best fit through [projection]. To see what this means, let’s quickly assume a toy example to work with: assume we have three points, $$(1, 1), (2, 2)$$ and $$(3, 2)$$, as shown below.

```python
import matplotlib.pyplot as plt
plt.use.style(“seaborn”)

data = [(1, 1), (2, 2), (3, 2)]
plt.plot(point) for point in data
plt.xlabel(“x”); plt.ylabel(“y”)
plt.show()

# Save and insert figure
```

As we can see, the three points do not form a single line. Therefore, it’s time for some regression. Let’s assume that this line is defined by $$y = ax + b$$. The system of equations which we will attempt to solve looks as follows:

$$\begin{cases} a + b = 1 \\ 2a + b = 2 \\ 3a + b = 2 \end{cases}$$

Or if you prefer the vector-matrix representation as I do,

$$\underbrace{\begin{pmatrix} 1 & 1 \\ 2 & 1 \\ 3 & 1 \end{pmatrix}}_{A} \underbrace{\begin{pmatrix} x \\ 1 \end{pmatrix}}_{x} = \underbrace{\begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix}}_{y}$$ 

This system, remind ourselves, does not have a solution because we have geometrically observed that no straight line can pass through all three points. What we can do, however, is find a projection of the vector $$y$$ onto matrix $$A$$ so that we can identify a solution that is closest to $$y$$, which we shall denote as $$\hat{y}$$. As you can see, this is where all the linear algebra kicks in.

## Deriving the Projection Formula

Let’s start by thinking about $$\hat{y}$$, the projection of $$y$$ onto $$A$$. After some thinking, we can convince ourselves that $$\hat{y}$$ is the component of $$y$$ that lives within the column space of $$A$$, and that $$y - \hat{y}$$ is the error component of $$y$$ that lives outside the column space of $$A$$. From this, it follows that $$y - \hat{y}$$ is orthogonal to $$C(A)$$, since any non-orthogonal component would have been factored into $$\hat{y}$$. Concretely,

$$A^{T}(y - \hat{y}) = 0$$

since the transpose is an alternate representation of the dot product. We can further specify this equation by using the fact that $$\hat{y}$$ can be expressed as a linear combination of the columns of $$A$$. In other words, 

$$A^{T}(y - A\hat{x}) = 0 \tag{1}$$

where $$\hat{x}$$ is the solution to the system of equations represented by $$Ax = \hat{y}$$. Let’s further unpackage (1) using matrix multiplication.

$$A^{T}y - A^{T}A\hat{x} = 0$$

Therefore, 

$$A^{T}y = A^{T}A\hat{x}$$

We finally have a formula for $$\hat{x}$$:

$$\hat{x} = (A^{T}A)^{-1}A^{T}y \tag{2}$$

Let’s remind ourselves of what $$\hat{x}$$ is and where we were trying to get at with projection in the context of regression. We started off by plotting three data points, which we observed did not form a straight line. Therefore, we set out to identify the line of best fit by expressing the system of equations in matrix form, $$Ax = y$$, where $$x = (a, b)^{T}$$. But because this system does not have a solution, we ended up modifying the problem to $$Ax = \hat{y}$$, since this is as close as we can get to solving an otherwise unsolvable system. So that’s where we are with equation (2): a formula for $$\hat{x}$$, which contains the parameters that define our line of best fit. Linear regression is now complete. 

# Matrix Calculus

There are other ways to derive the regression formula in (2). In this section, we will use matrix calculus as a tool to approach the linear regression problem. 

## Least Means Squared

If you think about it, there are many standards we can use to construct the line of best fit. 
