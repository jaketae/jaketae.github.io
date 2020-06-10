---
title: Contributing Open Source
toc: true
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

The main problem was that I was rushing myself. At one point, I tried a bunch of commands without knowing what I was really doing, and ultimately ended up creating a pull request to the upstream repository containing over a few hundred file modifications. Of course, it wasn't me who had made these modification: they were made by other developers working simulataneously on their own branches, which had been merged to upstream master. In retrospect, this happened because I had pushed a commit after pulling from the upstream instead of merging or rebasing my fork. When the pull request was made, I found myself panicking, struggling to understand what I had done but more embarassed to see a stupid pull request made public on GitHub. Of course, I promptly closed the request, but I was both surprised and abashed at the same time. This embarassing episode taught me a number of things. First, to really start contributing to open source, I would have to learn Git. Learn it properly. Second, open source contribution was definitely doable: had my PR been a good PR, I would not have closed it, and it would have been reviewed by members and authors of the repository, and hopefully even merged to master. With these in mind, I continued studying Git and finally made my first pull request that week.

After a few more days, I finally felt more comfortable with my knowledge of Github. This became my default workflow:

```bash
git checkout master
git fetch upstream
git rebase upstream/masterr
git checkout -b new-branch
cd some-directory
sublime .
git status
git add .
git commit -m "fix: some great commit message"
git push --set-upstream origin new-branch
```

Now, if I wanted to delete a branch for some reason, I could simply do the following:

```bash
git checkout master
git branch -d new-branch
git push origin --delete new-branch
```

These commands would effectively delete the branch both from the remote and local repository. 

These commands are obviously very simple and in no way provides a good introduction or crash course on Git commands--they are left here mostly for my own personal reference. Nonetheless, they are very simple commands that might as well help just about any student trying to get started with Git and open source contributions.  

# Ways to Contribute

There are many ways one can contribute to open source. In this section, I detail few of these ways by categorizing my own contributions I have made thusfar into three discrete groups.

## Starting Slow with Docstrings

Call me audacious, but my target open source projects were big, popular modules such as `tensorflow`, `keras`, and `pandas`. Intuitively, one would think that these repositories are most difficult to contribute to since they would presumably be the most well-maintained given the sheer number of users. This proposition is true to an extent, but it fails to consider the flip side of the coin: because these projects are massive, there is always constant room for improvement. Small repositories with just a few folders and files might be easier to maintain, and hence contain lesser lines that require fixing or refactoring. Big, popular projects are also highly well-maintained, but because of the sheer amount of features being rolled out and the number of pull requests being made, chances are there is always going to be room for contribution. 

At first, looking at the long lines of complicated code was rather intimidating because I had little understanding of the internal clockwork of the module. Therefore, I found it easier to work on editing the docstrings, which are multi-line comments that typically explain what a particular function does, its arguments, and return type. These comment appear on the official documentation that explains the API, which is why it is important to provide a clear, well-written description of the function. I decided to start slow by taking a look at the docstrings for each function and developing a gradual intuition of how one module interacts with another, how the interplay of various hidden functions can achieve a particular functionality, etc. Correcting mistakes or adding elaborations to docstrings still requires one to understand each function, but it also requires one to be a good presentor of information and a keen reviewer capable of detecting deficiencies in explanation.

A lot of my pull requests fall under this category because, as a novice open source contributor, I still don't have a solid grasp of how TensorFlow works, for example. At best, I'm only familar with the `tensorflow.keras` API, because this is the module that I use the most frequently. Therefore, my strategy was to focus on docstrings for other `tensorflow` sub-modules while reviewing the code for `tensorflow.keras`. 

## Community Translations

Another huge portion of my work revolves around translating [tutorials](https://www.tensorflow.org/tutorials/) and [guides](https://www.tensorflow.org/guides/). I found this work to be incredibly rewarding for two reasons: first, I found myself learning tremdously a lot from trying to understand the underlying concepts and code. The first tutorial I translated was on the topic of conditional variational autoenoders, and I ran into references to familiar concepts such as multivariate Gaussians, KL divergence, and more. To translate, I had to make sure that I had a solid grasp of the material, so I spent some time brushing up on rusty concepts and learning gaps in knowledge that had to be filled. Secondly, I love translating because I feel like translating is a way of disseminating knowledge. It feels incredibly rewarding to think that some other budding developer will read my translation of how to implement [DeepDream](https://www.tensorflow.org/tutorials/generative/deepdream) or perform a [Fast Gradient Sign Method-based attack](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm) on a neural network to do more amazing things with the knowledge in hand.  

In retrospect, this is another perk of trying to contribute to a huge open source project like TensorFlow. Small projects typically do not have multi-language support, and it is only when projects grow big enough and attract a substational number of users that the authors and members decide to scale it to cater to a wider global audience. I'm happy to be given the opportunity to take part in this communal endeavor, and I only wish to do more.

## Optimization and Refactoring

The last category, of coure, has to do with editing the actual source code. This is the most technical part as it requires knowledge of not only general coding, but also the ins and outs of the library. I have made only a few contributions that fall under this category, but it is when these types of pull requests are approved and merged that I feel the most excitement. 

One of my first contributions involved the `pandas` library. It was a very minor edit to the code that made use of `defaultdict` instead of the default Python dictionary. However, making that pull request---and eventually pushing it through until merge---taught me a lot about optimization pull requests. When my PR was made public, one of the reviewers requested me to run a test to prove just how optimal my code would be. I was glad that my PR was reviewed, but at the same time daunted by this task. After a bit of research, I came up with my first primitive approach:

```python
test_dict = {}
test_defaultdict = defaultdict(dict)
dict_time = []
defaultdict_time = []

for _ in range(10000):
    start_time = time.time()
    test_lst = [test_dict.get(x, {}) for x in range(1000)]
    dict_time.append(time.time() - start_time)

for _ in range(10000):
    start_time = time.time()
    test_lst = [test_defaultdict[x] for x in range(1000)]
    defaultdict_time.append(time.time() - start_time)

print(sum(dict_time)/len(dict_time))
print(sum(defaultdict_time)/len(defaultdict_time))
```

This returned the following result:

```python
0.000193774962425 
9.86933469772e-05 
```

This proved that the performance boost was indeed there. To this point, the reviewers agreed, yet the guided me to use the `timeit` feature instead to prove my point with a plausible input. This was my second attempt at the tasks given their feedback:

```python
FACTOR, DIM = 0.1, 10000

data = {f'col {i}': 
        {f'index {j}': 'val' for j in random.sample(range(DIM), int(FACTOR * DIM))} 
        for i in range(DIM)} # randomly populate nested dictionary

defaultdict_data = defaultdict(dict, data) # equivalent data in defaultdict

def _from_nested_dict(data): # original method
    new_data = {}
    for index, s in data.items():
        for col, v in s.items():
            new_data[col] = new_data.get(col, {})
            new_data[col][index] = v
    return new_data

def _from_nested_dict_PR(data): # PR
    new_data = defaultdict(dict)
    for index, s in data.items():
        for col, v in s.items():
            new_data[col][index] = v
    return new_data
```

Running a `timeit` returned the following result:

```
>>> %timeit _from_nested_dict(data)
6.99 s ± 33.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
>>> %timeit _from_nested_dict_PR(defaultdict_data)
4.88 s ± 32 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

This was better, as it directly proved that my PR optimized performance in the context of the function in question, instead of dealing with the more general question of `dict` versus `defaultdict`. 

After this PR was merged into master, I got a better sense of what optimizing code entailed and what the expectations were from the point of view of the project maintainers. It was also great to learn how to tangibly measure code performance via `timeit`, which is something that I did not know about previousely.

Optimization is not the only way to contribute to source code: my attempts have included adding robust `dtype` check for arguments, adding a TensorFlow model such as `resnext` to the official list of models, and a few more. Making contributions to source code definitely feels more difficult than editing docstrings, for instance, but I'm hoping that with more practice and experience, spotting errors and rooms for improvement will come more naturally.

# Conclusion

Making open source contributions is not easy, but it is also not impossible. In fact, I would say that anyone who is determined to parse through code and comments can make a good pull request, however small the edit may be. I'm definitely in the process of learning about open source contributions, especially when it comes to reviewing code and adding features, but I hope to contribute what I can as a student and individual developer. After all, it feels good to know that there is at least a drop of my sweat involved in the statement `import tensorflow as tf`.

If anything, it is my hope that this post will also inspire you to contribute to open source; after all, collective intelligence is what makes open source projects so cool.
