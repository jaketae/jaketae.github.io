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


# Constructing the Stochastic Matrix

Now that we have reviewed some underlying concepts, perhaps it is time to apply our knowledge to a concrete example. Before we move on, I recommend that you check out [this post] I have written on the Markov process, just so that you are comfortable with the material to be presented in this section. 

In this post, we turn our attention to the game of Chutes and Ladders, which is an example of a Markov process which demonstrates the property of "[memorylessness]." This simply means that the progress of the game depends only on the players' current positions, not where they were or how they got there. A player might have ended up where they are by taking a ladder or by performing a series of regular dice rolls. In the end, however, all that matters is that the players eventually hit the hundredth cell. 

<figure>
	<img src="/assets/images/chutes-and-ladders.png">
	<figcaption>Figure 1: Chutes and Ladders game</figcaption>
</figure>

To perform a Markov chain analysis on the Chutes and Ladders game, it is first necessary to convert the information presented on the board as a [stochastic matrix]. How would we go about this process? Let's assume that we start the game at the $$0$$th cell by rolling a dice. There are six possible events, each with probability of $$1/6$$. More specifically, we can end up at the index numbers 38, 2, 3, 14, 5, or 6. In other words, at position 0, 

$$P(X = x \vert C = 0) = 
\begin{cases}
\ \frac 16 & \{x \vert x \in 38, 2, 3, 14, 5, 6\} \\[2ex]
\ 0 & \{x \vert x \notin 38, 2, 3, 14, 5, 6\}
\end{cases}$$

where $$C$$ and $$X$$ denote the current and next position of the player on the game board, respectively. We can make the same deductions for other cases where $$C = 1 \ldots 100$$. We are thus able to construct a 101-by-101 matrix representing the transition probabilities of our Chutes and Ladders system, where each column represents the system at a different state, *i.e.* the $$j$$th entry of the $$i$$th column vector represents the probabilities of moving from cell $$i$$ to cell $$j$$. To make this more concrete, let's consider a program that constructs the stochastic matrix $$T1$$, without regards to the chutes and ladders for now. 

```python
import numpy as np

def board_generator(dim=101):
"""Returns a 2D numpy representation of the game board."""

    # Dictionary of chutes and ladders
    # To be added 
    chutes_ladders = {...}

    # Initialize ndarray of zeros
    T1 = np.zeros((dim, dim))

    # Assign probability 1/6 to appropriate entries
    for i in range(101):
        T1[i + 1:i + 1 + 7] = 1. / roll

    return T1

```

The indexing is key here: for each column, $$[i + 1, i + 7)$$th rows were assigned the probability of $$1/6$$. Let's say that a player is in the $$i$$th cell. Assuming no chutes or ladders, a single roll of a dice will place him at one of the cells from $$(i + 1)$$ to $$(i + 6)$$; hence the indexing as presented above. So now we're done!

... or not quite. 

Things get a bit more complicated once we throw the chutes and ladders into the mix. To achieve this, we first build a dictionary containing information on the jump from one cell to another. In this dictionary, the keys correspond to the original position; the values, the index of the cell after the jump, either through a chute or a ladder.

```python
chutes_ladders = {
1: 38, 4: 14, 9: 31, 16: 6, 21: 42,
28: 84, 36: 44, 47: 26, 49: 11, 51: 67,
56: 53, 62: 19, 64: 60, 71: 91, 80: 100,
87: 24, 93: 73, 95: 75, 98: 78
}
```

For example, ```1: 38``` represents the first ladder on the game board, which moves the player from the first cell to the thirty eighth cell. 

To integrate this new piece of information into our code, we need to build a permutation matrix that essentially "shuffles up" the entries of the stochastic matrix $$T1$$ in such a way that the probabilities can be assigned to the appropriate entries. For example, $$T1$$ does not reflect the fact that getting a 1 on a roll of the dice will move the player up to the thirty eighth cell; it supposes that the player would stay on the first cell. The new permutation matrix $$T2$$ would adjust for this error by reordering $$T1$$. For an informative read on the mechanics of permutation, refer to this [explanation on Wolfram Alpha]. 

```python
# Initialize ndarray of zeros
T2 = np.zeros((dim, dim))

# ndarray of 101 elements
# If i in chutes_ladders, i is replaced with corresponding value
index_lst = [chutes_ladders.get(i, i) for i in range(101)]

# Permutation matrix
T2[index_lst, range(101)] = 1
```

Let's perform a quick sanity check to verify that $$T2$$ contains the right information on the first ladder, namely the entry ```1: 38``` in the ```chutes_ladders``` dictionary.

```python
>>> print(T2[:, 1])

>>> [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0.]
```

Notice the $$1$$ in the $$38$$th entry hidden among a haystack of 100 $$0$$s! This result tells us that $$T2$$ is indeed a permutation matrix whose multiplication with $$T1$$ will produce the final stochastic vector that correctly enumerates the probabilities encoded into the Chutes and Ladders game board. Here is our final product:

```python
import numpy as np
import matplotlib.pyplot as plt

def board_generator(dim=101):

    chutes_ladders = {1: 38, 4: 14, 9: 31, 16: 6, 21: 42,
                      28: 84, 36: 44, 47: 26, 49: 11, 51: 67,
                      56: 53, 62: 19, 64: 60, 71: 91, 80: 100,
                      87: 24, 93: 73, 95: 75, 98: 78}

    T1 = np.zeros((dim, dim))
    T2 = np.zeros((dim, dim))

    for i in range(101):
        T1[i + 1:i + 1 + 7] = 1. / roll

    index_lst = [chutes_ladders.get(i, i) for i in range(101)]

    T2[index_lst, range(101)] = 1

    T = T2 @ T1

    return T
```

We can visualize the stochastic matrix $$T$$ using the ```matplotlib``` package. 

```python
stochastic_mat = board_generator()
plt.matshow(stochastic_mat)
plt.show()
```
This produces the following output, which is a visualization of our stochastic matrix. 

<figure>
	<img src="/assets/images/game-board.png">
	<figcaption>Figure 2: Visualization of the stochastic matrix</figcaption>
</figure>

Now that we have a stochastic matrix to work with, we can finally perform more mathematical analyses to better understand its structure. 

# Eigendecomposition

At this point, let's remind ourselves of the end goal. Since we have successfully built a stochastic matrix, all we have to do is to set some initial starting vector $$x_0$$ and perform iterative matrix calculations. In recursive form, this statement can be expressed as follows:

$$x_{n+1} = Tx_n = T^{n+1}x_0$$

At a glance, this approach is the exact one which we took in the [previous post] with the mini PageRank algorithm. However, we will see here that performing a [eigendecomposition] of the stochastic matrix will drastically improve the calculation process, and with that, a reduction in our program's runtime. 

Eigendecomposition refers to a specific method of factorizing a matrix in terms of its eigenvalues and eigenvectors. There are many ways to understand this fascinating operation, but I find it most intuitive to start by considering the operation of matrix multiplication. Let's clear out the notation first: let $$A$$ be the matrix of interest, $$S$$ a matrix whose columns are each an eigenvector of $$A$$, and $$\Lambda$$, a matrix whose diagonal entries are each the corresponding eigenvalues of $$S$$. 

First, we begin by considering the result of multiplying $$A$$ and $$S$$. What would the result be? If we consider matrix as a repetition of matrix-times-vector operations, we can yield the following result.

$$AS = A \cdot \begin{pmatrix} \vert & \vert &        & \vert \\ s_1 & s_2 & \ldots & s_n \\ \vert & \vert &        & \vert \end{pmatrix} = \begin{pmatrix} \vert & \vert &        & \vert \\ As_1 & As_2 & \ldots & As_n \\ \vert & \vert &        & \vert \end{pmatrix}$$

But recall that $$s$$ are eigenvectors of $$A$$, which necessarily implies that

$$As_n = \lambda s_n$$

Therefore, the result of $$AS$$ can be rearranged and unpacked in terms of $$\Lambda$$:

$$ \begin{pmatrix} \vert & \vert &        & \vert \\ As_1 & As_2 & \ldots & As_n \\ \vert & \vert &        & \vert \end{pmatrix} = \begin{pmatrix} \vert & \vert &        & \vert \\ \lambda_1 s_1 & \lambda_2 s_2 & \ldots & \lambda_n s_n \\ \vert & \vert &        & \vert \end{pmatrix}$$
$$ = \begin{pmatrix} \vert & \vert &        & \vert \\ s_1 & s_2 & \ldots & s_n \\ \vert & \vert &        & \vert \end{pmatrix} \cdot \begin{pmatrix} \lambda_1 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & \lambda_n \end{pmatrix} = S \Lambda $$

In short,

$$AS = S \Lambda$$

Right-multiplying both sides by $$S^{-1}$$ produces the following:

$$ASS^{-1} = A = S \Lambda S^{-1}$$

Therefore, we have $$A = S \Lambda S^{-1}$$, which is the formula for eigendecomposition of a matrix. 

# Powers of the Stochastic Matrix

One of the beauties of eigendecomposition is that it allows us to compute matrix powers very easily. Concretely, 

$$A^n = {(S \Lambda S^{-1})}^n = (S \Lambda S^{-1}) \cdot (S \Lambda S^{-1}) \dots (S \Lambda S^{-1}) = S \Lambda^n S^{-1}$$

Because $$S$$ and $$S^{-1}$$ nicely cross out, all we have to compute boils down to $$\Lambda^n$$! But the good news doesn't stop here: because $$\Lambda$$ is a diagonal matrix, $$\Lambda^n$$ is simply the matrix with the diagonal, nonzero entries of $$\Lambda$$ exponentiated by $$n$$:

$$\Lambda^n = \begin{pmatrix} \lambda_1 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & \lambda_n \end{pmatrix}^{n} = \begin{pmatrix} \lambda_1^n & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & \{lambda_n}^n \end{pmatrix}$$

Calculating powers of a matrix in its eigendecomposed form, therefore, is a computationally lightweight task. 




[previous post]: https://jaketae.github.io/blog/math/pagerank-and-markov/

[this post]: https://jaketae.github.io/blog/math/pagerank-and-markov/

[defined]: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors

[stochastic matrix]: http://mathworld.wolfram.com/StochasticMatrix.html

[explanation on Wolfram Alpha]: http://mathworld.wolfram.com/PermutationMatrix.html

[memorylessness]: https://en.wikipedia.org/wiki/Markov_property

[characteristic polynomial]: http://mathworld.wolfram.com/CharacteristicPolynomial.html

[eigendecomposition]: https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix