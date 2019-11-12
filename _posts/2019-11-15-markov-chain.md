---
title: "Markov Chain and Chutes and Ladders"
last_modified_at: 2019-11-16 4:42:00 +0000
categories:
  - blog
  - math
tags:
  - study
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

In a [previous post], we briefly explored the notion of Markov chains and their application to Google's PageRank algorithm. Today, we will attempt to understand the Markov process from a more mathematical standpoint by meshing together the concept of eigenvectors and Monte Carlo simulations. 

# Revisiting Eigenvectors

In linear algebra, an eigenvector of a linear transformation is roughly [defined] as follows:

> a nonzero vector that is mapped by a given linear transformation onto a vector that is the scalar multiple of itself

This definition, while seemingly abstract and cryptic, distills down into a simple equation when written in matrix form:

$$Ax = \lambda x$$

Here, $$A$$ denotes the matrix representing a linear transformation; $$x$$, the eignevector; $$\lambda$$, the scalar value that is multiplied onto the eigenvector. Simply put, an eigenvector $$x$$ of a linear transformation is one that is--allow me to use this term in the loosest sense to encompass positive, negative, and even imaginary scalar values--"stretched" by some factor $$\lambda$$ when the transformation is applied, *i.e.* multiplied by the matrix $$A$$ which maps the given linear transformation. 

The easiest example I like to employ to demonstrate this concept is the identity matrix $$I$$. For the purpose of demonstration, let $$a$$ be an arbritrary vector $$(x, y, z)^{T}$$ and $$I$$ the three-by-three identity matrix. Multiplying $$a$$ by $$I$$ produces the following result:

$$Ia = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \cdot \begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} x \\ y \\ z \end{pmatrix}$$

The result is unsurprising, but it reveals an interesting way of understanding $$I$$: identity matrices are a special case of diagonalizable matrices whose eigenvalues are 1. Because the multiplying any arbitrary vector by the identity matrix returns the matrix itself, all vectors in the dimensional space can be considered an eigenvector to the matrix $$I$$, with $$\lambda$$ = 1. 

A formal way to calculate eigenvectors and eigenvalues can be derived from the equation above. 

$$Ax = \lambda x$$

$$Ax - \lambda x = 0$$

$$(A - \lambda I)x = 0$$

Since $$x$$ is assumed as a nonzero vector, we can deduce that the matrix $$(A - \lambda I)$$ is a singular matrix with a nontrivial null space. In fact, the vectors in this null space are precisely the eigenvectors that we are looking for. Here, it is useful to recall that the a way to determine the singularity of a matrix is by calculating its determinant. Using these set of observations, we can modify the equation above to the following form:

$$det(A - \lambda I) = 0$$

By calculating the determinant of $$(A - \lambda I)$$, we can derive the [characteristic polynomial], from which we can obtain the set of eigenvectors for $$A$$ representing some linear transformation $$T$$. 


# Chutes and Ladders game

Now that we have reviewed some underlying concepts, perhaps it is time to apply our knowledge to a concrete example. Before we move on, I recommend that you check out [this post] I have written on the Markov process, just so that you are comfortable with the material to be presented in this section. 

In this post, we turn our attention to the game of Chutes and Ladders, which is an example of a Markov process which demonstrates the property of "[memorylessness]." This simply means that the progress of the game depends only on the players' current positions, not where they were or how they got there. A player might have ended up where they are by taking a ladder or by performing a series of regular dice rolls. In the end, however, all that matters is that the players eventually hit the hundredth cell. 

<figure>
	<img src="/assets/images/chutes-and-ladders.png">
	<figcaption>Figure 2: Chutes and Ladders game</figcaption>
</figure>

To perform a Markov chain analysis on the Chutes and Ladders game, it is first necessary to convert the information presented on the board as a [stochastic matrix]. How would we go about this process? Let's assume that we start the game at the $$0$$th cell by rolling a dice. There are six possible events, each with probability of $$1/6$$. More specifically, we can end up at the index numbers 38, 2, 3, 14, 5, or 6. In other words, at position 0, 

$$P(X \vert C = 0) = 
\begin{cases}
\ 1/6 & \text{if X \in \{38, 2, 3, 14, 5, 6\}} \\[2ex]
\ 0 & \text{if X \notin \{38, 2, 3, 14, 5, 6\}}
\end{cases}

where $$C$$ denotes the current position of the player on the game board. We can make the same deductions for other cases where $$C = 1 \ldots 100$$. We are thus able to construct a 101-by-101 matrix representing the transition probabilities of our Chutes and Ladders system, where each column represents the system at a different state, *i.e.* the $$j$$th entry of the $$i$$th column vector represents the probabilities of moving from cell $$i$$ to cell $$j$$. To make this more concrete, let's consider a program that constructs the transition matrix $$T$$, without regards to the chutes and ladders for now. 

```python
import numpy as np

def board_generator(roll=6, dim=101):
"""Returns a 2D numpy representation of the game board."""

    # Dictionary of chutes and ladders
    CHUTES_LADDERS = {...}

    # Initialize with all zero entries
    T = np.zeros((dim, dim))

    # Assign probability 1/6 to appropriate entries
    for i in range(101):
        T[i + 1:i + 1 + roll, i] = 1. / roll

    return T

```












[previous post]: https://jaketae.github.io/blog/math/pagerank-and-markov/

[this post]: https://jaketae.github.io/blog/math/pagerank-and-markov/

[defined]: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors

[stochastic matrix]: http://mathworld.wolfram.com/StochasticMatrix.html

[memorylessness]: https://en.wikipedia.org/wiki/Markov_property

[characteristic polynomial]: http://mathworld.wolfram.com/CharacteristicPolynomial.html