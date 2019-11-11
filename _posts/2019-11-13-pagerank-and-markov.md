---
title: "Understanding PageRank"
last_modified_at: 2019-11-13 4:42:00 +0000
categories:
  - blog
  - math
tags:
  - study
---

# The Google Search Engine

Google is the most popular search engine in the world. It is so popular that the word "Google" has been added to the Oxford English Dictionary as a proper verb, denoting the act of searching on Google. 

While Google's success as an Internet search engine might be attributed to a plethora of factors, the company's famous PageRank algorithm is undoubtedly a contributing factor behind the stage. The PageRank algorithm is a method by which Google ranks different pages on the world wide web, displaying the most relevant and important pages on the top of the search result when a user inputs an entry. Simply put, PageRank determines which websites are most likely to contain the information the user is looking for and returns the most optimal search result. 

# Network Graph Representation

While the nuts and bolts of this algorithm may appear complicated--and indeed it is--the underlying concept is surprisingly intuitive: the relevance or importance of a page is determined by the number of hyperlinks going to and from the website. Let's hash out this proposition by creating a miniature version of the Internet. In our miniature world, there are only five websites, represented as nodes on a network graph. Below is a simple representation created using Python and the Networkx package. 

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
links = [("A", "B"), 
		 ("B", "A"), 
		 ("B", "C"), 
		 ("C", "A"), 
		 ("C", "B"), 
		 ("C", "E"), 
		 ("D", "A"), 
		 ("E", "D"), 
		 ("E", "B"), 
		 ("E", "C")]

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
	<figcaption>Representation of a miniature world wide web</figcaption>
</figure>

How is this a model of the Internet? Well, as stupid as it seems, the network graph contains all the information necessary for our preliminary analysis: namely, hyperlinks from one page to another. Let's take node D as an example. The pointed edges indicate that page E contains a link tot page D, and that page D contains another link that then redirects the user to page A. Interpreted in this fashion, the graph shows which pages have a lot of incoming and outgoing reference links. 

Why are hyperlinks important for the PageRank algorithm, you might ask. A useful intuit might be that websites with a lot of incoming references are likely to be influential sources, often written by prominent individuals. This analysis is certainly the case in the field of academics, where works of literature that are frequently cited quickly gain clout and reach an established position in the given discipline. Another reasoning is that hyperlinks tell us where a user is most likely to end up in. Take the extreme example of an isolated node, where there are no outgoing or ingoing links to the website. It is unlikely that a user will end up on that website, as opposed to a popular page with a spiderweb of edges. Then, it would make sense for the PageRank algorithm to display that website on top of the one represented by an isolated node. 

# Random Walk

Suppose we want to know where a user is most likely to end up in after a given search. This process is referred to as a "random walk" because we are essentially trying to 





