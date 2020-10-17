---
title: Hidden Markov Models
mathjax: true
toc: true
categories:
  - study
tags:
  - machine_learning
  - nlp

---

It's been a while since I've uploaded my last post, and it certainly feels great again to be writing on this blog. In today's long overdue post, we'll explore an interesting model with applications in NLP: Hidden Markov Models, or HMMs for short. 

Before moving on any further, full disclosure that this post is heavily based on [this video](https://www.youtube.com/watch?v=fX5bYmnHqqE) by [ritvikmath](https://www.youtube.com/channel/UCUcpVoi5KkJmnE3bvEhHR0Q), which is one of the many math channels on YouTube that I enjoy watching. I borrowed the example in his video to describe the theory behind HMMs in this post. 

Without further ado, let's get started. 

# Setup

Perhaps it's best to start off this post by explaining a hypothetical scenario in which HMMs might come in handy. Generally speaking, HMMs are employed in situations where there are a set of latent variables that affect our observed data. We have no way of directly observing these latent variables, but we know that there is some relationship between observed and latent variables. This is the "Hidden" portion of HMMs.

The "Markov" portion of HMMs comes from the fact that we assume the [Markov property](https://en.wikipedia.org/wiki/Markov_property)---sometimes referred to as the memoryless property---for those latent variables. This simply means that there are some fixed quantities that determine the probability a transition from one state to another, *i.e.* the transition from one state to another only depends on the current state itself. All of this should be fairly familiar from [previous posts on this blog](https://jaketae.github.io/tags/#markov-chain), so I recommend that you check some of that out if you think you'll need some refreshers on things like Markov chains. 

## Example

So here's a concrete example. Assume that we have a friend, let's say Jake, who is always in either one of two moods on any given day: happy or sad. Whether he is happy or sad only depends on his mood yesterday. The problem is that we don't know the transition probabilities from happy to happy, happy to sad, sad to happy, or sad to sad; indeed, we are left completely clueless. However, there is one indicator we can use to extrapolate Jake's mood, and that is the color of the shirt he wears, which is always one of three colors: red, green, or blue. As you may be able to guess







 



