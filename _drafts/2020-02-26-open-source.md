---
title: Contributing Open Source
toc: true
date: 2020-03-04
categories:
  - study
  - development
tags:
  - update


---

Programming is difficult but fun. Or maybe it's the other way around. Either way, any  developer would know that external libraries are something that makes programming so much easier. As a Pythonista with an interest in deep learning, I cannot image a world without essential libraries like `numpy`, `matplotlib`, `pandas`, or `tensorflow`. It simply be  impossible, at least for me, to achieve everything from scratch without using these libraries. 

It is only a few days ago that it dawned to me that all of the modules I frequently use are open source, publicly available and accessible on GitHub. This not only meant that I could take a look at what happens under the hood when I use import statements or call internal functions, but also that I could actively contribute to maintaining and improving these libraries myself. This realization came to me roughly at a time when I was looking into a program called [Google Summer of Code](https://summerofcode.withgoogle.com), which provides a systematic platform for students enrolled in tertiary institutions to contribute to a specific open source project alongside a designated mentor. Unfortunately, I was not eligible for this program due to my being on an authorized leave of absence; however, I decided that it would be great to contribute to an open source library and decided to embark on this journey nonetheless. And of course, with that decision came a number of different hurdles... and a sweet taste of fruition at last.

# Learning Git from Mistakes

It soon became immediately obvious that I had to learn how to use Git. I had a faint idea of what Git and GitHub were, since I had to use GitHub to maintain this very blog. However, I never used the command line for any commits; instead, I used a handy little application called [GitHub Desktop](https://desktop.github.com/) that enabled novice users like me to interact with Git and use its features without going through the complexities of the command line interface. The obvious benefit of relying on this applicaiton is that things are so simple: just work on the project, save your files, fire up the application, and it would automatically pull up all the documents that were modified, ready to be pushed. The downside of this approach, however, was that I had virtually zero working knowledge of Git and version control. Unsurprisingly in retrospect, this turned out to be a huge deficiency when I decided to attempt contributing to open source. For the next two days, I went on a Googling spree, learning about basic Git commands and jargons such as origin, remote, master, branch, fork, and clone, to name a few. And also because I believe that learning is best done by doing, I forked and cloned actual open source repositories to try out Git commands myself.

The main problem was that I was rushing myself. At one point, I tried a bunch of commands without knowing what I was really doing, and ultimately ended up creating a pull request to the upstream repository containing over a few hundred file modifications. In retrospect, this happened becasue I had pushed a commit after pulling from the upstream instead of merging or rebasing my fork. When the pull request was made, I found myself panicking, struggling to understand what I had done but more embarassed to see a stupid pull request made public on GitHub. Of course, I promptly closed the request, but I was both surprised and abashed at the same time. 

This embarassing episode taught me a number of things. First, to really start contributing to open source, I would have to learn Git. Learn it properly. Second, open source contribution was definitely doable: had my PR been a good PR, I would not have closed it, and it would have been reviewed by members and authors of the repository, and hopefully even merged to master. With these in mind, I continued studying Git and finally made my first pull request that week.

# Starting Slow with Docstrings









 

