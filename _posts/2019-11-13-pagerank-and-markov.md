---
title: "Understanding PageRank"
last_modified_at: 2019-11-13 4:42:00 +0000
categories:
  - blog
  - math
tags:
  - study
---

Google is the most popular search engine in the world. It is so popular that the word "Google" has been added to the Oxford English Dictionary as a proper verb, denoting the act of searching on Google. 

While Google's success as an Internet search engine might be attributed to a plethora of factors, the company's famous PageRank algorithm is undoubtedly a contributing factor behind the stage. The PageRank algorithm is a method by which Google ranks different pages on the world wide web, displaying the most relevant and important pages on the top of the search result when a user inputs an entry. Simply put, PageRank determines which websites are most likely to contain the information the user is looking for and returns the most optimal search result. 

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

Running this block results in the network graph below:

<figure>
	<img src="/assets/images/graph.png">
	<figcaption>Representation of a miniature world wide web.</figcaption>
</figure>


