---
title: "Markov Chain and Chucks and Ladders"
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

> a nonzero vector that is mapped by a given linear transformation onto a vector that is the scalar multiple of the original vector

This definition, while seemingly abstract and convolouted, distills down into a simple equation when written in matrix form:

$$Ax = \lambda x$$

Here, $$A$$ denotes the matrix representing a linear transformation; $$x$$, the eignevector; $$\lambda$$, the scalar value that is multiplied onto the eigenvector. Simply put, an eigenvector $$x$$ of a linear transformation is one that is "stretched" or "shrunk" by some factor $$\lambda$$ when the transformation is applied, *i.e.* multiplied by the matrix $$A$$ which maps the given linear transformation. 




Now that we have reviewed the underlying concept, perhaps it is time to apply our knowledge with an example. Here is the transition matrix $$P$$ introduced in the previous post on the PageRank algorithm. 

$$P = \begin{pmatrix} 0 & 1/2 & 1/3 & 1 & 0 \\ 1 & 0 & 1/3 & 0 & 1/3 \\ 0 & 1/2 & 0 & 0 & 1/3 \\ 0 & 0 & 0 & 0 & 1/3 \\ 0 & 0 & 1/3 & 0 & 0 \end{pmatrix}$$







[previous post]: https://jaketae.github.io/blog/math/pagerank-and-markov/

[defined]: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors