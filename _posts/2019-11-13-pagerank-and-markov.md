---
title: "Understanding PageRank"
last_modified_at: 2019-11-13 4:42:00 +0000
categories:
  - blog
  - math
tags:
  - study
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

Google is the most popular search engine in the world. It is so popular that the word "Google" has been added to the Oxford English Dictionary as a proper verb, denoting the act of searching on Google. 

While Google's success as an Internet search engine might be attributed to a plethora of factors, the company's famous PageRank algorithm is undoubtedly a contributing factor behind the stage. The PageRank algorithm is a method by which Google ranks different pages on the world wide web, displaying the most relevant and important pages on the top of the search result when a user inputs an entry. Simply put, PageRank determines which websites are most likely to contain the information the user is looking for and returns the most optimal search result. 

# Network Graph Representation

While the nuts and bolts of this algorithm may appear complicated--and indeed they are--the underlying concept is surprisingly intuitive: the relevance or importance of a page is determined by the number of hyperlinks going to and from the website. Let's hash out this proposition by creating a miniature version of the Internet. In our microcosm, there are only five websites, represented as nodes on a network graph. Below is a simple representation created using Python and the Networkx package. 

```python
import networkx as nx
import matplotlib.pyplot as plt

# Initialize a directed graph
DG = nx.DiGraph()

# Create a list of webpages
pages = ["A", "B", "C", "D", "E"]

# Add nodes
DG.add_nodes_from(pages)

# Create a list of hyperlinks
# (X, Y) represents a directed edge from X to Y
links = [("A", "B"), ("B", "A"), 
("B", "C"), ("C", "A"), 
("C", "B"), ("C", "E"), 
("D", "A"), ("E", "D"), 
("E", "B"), ("E", "C")]

# Add edges
DG.add_edges_from(links)

# Draw graph
nx.draw(DG, with_labels = True)
plt.draw()
plt.show()

```

Running this block results in the following graph:

<figure>
	<img src="/assets/images/graph.png">
	<figcaption>Figure 1: Representation of a miniature world wide web</figcaption>
</figure>

How is this a model of the Internet? Well, as simple as it seems, the network graph contains all the pertinent information necessary for our preliminary analysis: namely, hyperlinks going from one page to another. Let's take node D as an example. The pointed edges indicate that page E contains a link to page D, and that page D contains another link that redirects the user to page A. Interpreted in this fashion, the graph indicates which pages have a lot of incoming and outgoing reference links. 

But all of this aside, why are hyperlinks important for the PageRank algorithm in the first place? A useful intuition might be that websites with a lot of incoming references are likely to be influential sources, often written by prominent individuals. This analysis is certainly the case in the field of academics, where works of literature that are frequently cited quickly gain clout and reach an established position in the given discipline. Another reasoning is that hyperlinks tell us where a user is most likely to end up on after browsing through returned search results. Take the extreme example of an isolated node, where there are zero outgoing and ingoing links to the website. It is unlikely that a user will end up on that webpage, as opposed to a popular site with a spiderweb of edges on a network graph. Then, it would make sense for the PageRank algorithm to display that website on top; the isolated node, the bottom. 

# Markov Chain

Suppose we want to know where a user is most likely to end up in after a given search. This process is often referred to as a "random walk" or a "stochastic process" because, as the name suggests, it describes a path after a succession of random steps on some mathematical space. While it is highly unlikely that a user visits a website, randomly selects one of the hyperlinks on the given page, and repeats the two steps above repeatedly, the assumption on randomness is what allows us to simulate a user's navigation of the Internet from the point of view of Markov chains, a stochastic model that describes a sequence of possible events, or states, in which the probability of each event is contingent only upon the previous state attained in the previous event. One good example of a Markov chain is the famous Chutes and Ladders game, in which the player's next position is dependent only upon their present position on the game board. For this reason, Markov chains are said to be "memoryless": in the Chutes and Ladders game, whether the player ended up in their current position by taking a ladder or a normal dice roll is irrelevant to the progress of the game after all. 

<figure>
	<img src="/assets/images/chutes-and-ladders.png">
	<figcaption>Figure 2: Chutes and Ladders game</figcaption>
</figure>

A salient characteristic of a Markov chain is that the probabilities of each event can be represented and calculated by simple matrix multiplication. The specifics of this mechanism will be a topic for another post, but intuitively speaking, there would be some transition matrix $$P$$ that represents probabilities, and some vector $$x_n$$ that denotes the $$n^{th}$$ state in the Markov chain. Then, multiplying this state vector by the transition matrix would yield $$Mx_n = x_{n+1}$$, where the new vector $$x_{n+1}$$ denotes the probability distribution in the $${n+1}^{th}$$ state in the Markov chain. The beauty behind the Markov chain is that the result of this multiplication operation, when iterated many times, converges to a stationary distribution vector regardless of where we started from, *i.e.* $$x_0$$. 

To make all of this more concrete, let's return back to our example of the Internet microcosm and the five websites. In order to apply a stochastic analysis on our model, it is first necessary to translate the network graph presented above into a transition matrix $$M$$ whose individual entries are nonnegative real numbers that denote some probability of change from one state to another. 

Here is the matrix representation of the network graph in our example:

$$P = \begin{pmatrix} 0 & 1/2 & 1/3 & 1 & 0 \\ 1 & 0 & 1/3 & 0 & 1/3 \\ 0 & 1/2 & 0 & 0 & 1/3 \\ 0 & 0 & 0 & 0 & 1/3 \\ 0 & 0 & 1/3 & 0 & 0 \end{pmatrix}$$

The column vector $$x_n$$ can be defined as 

$$x_n = \begin{pmatrix} P(A) \\ P(B) \\ P(C) \\ P(D) \\ P(E) \end{pmatrix}$$

where $$P(X)$$ denotes the probability that the user is browsing website (or node from a network graph's perspective) $$X$$ at state $$n$$. For instance, $$x_0 = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$ would be an appropriate vector representation of a state distribution in a Markov chain where a user began their random walk at website A. This will be our example. 

From these, how can we learn more more about $$x_1$$? $$Mx_0$$ would give us the answer:

$$Mx_0 = begin{pmatrix} 0 & 1/2 & 1/3 & 1 & 0 \\ 1 & 0 & 1/3 & 0 & 1/3 \\ 0 & 1/2 & 0 & 0 & 1/3 \\ 0 & 0 & 0 & 0 & 1/3 \\ 0 & 0 & 1/3 & 0 & 0 \end{pmatrix}$$ \cdot $$\begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

Notice that the result is just the first column of the transition matrix! 
 

















