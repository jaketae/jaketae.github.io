---
title: Scikit-learn Sprint
mathjax: false
toc: false
header:
  image: /assets/images/scikit_learn_logo_small.svg
categories:
  - development
  - study
tags:
  - update
---

Last week, I had the pleasure of joining the first ever virtual [Scikit-learn sprint](https://sites.google.com/view/nyc-2020-scikit-sprint/home), kindly organized by [Data Umbrella](https://www.dataumbrella.org) and [NYC PyLadies](https://nyc.pyladies.com). Although it was just a four-hour sprint held entirely online, it was a fantastic learning experience that helped me better understand collaborative workflow and the joy of contributing to open source. 

A salient feature of the Scikit-learn sprint is pair programming, where participants are grouped into teams consisting of two people. Each pair works as a team to address open issues, report new issues, or submit pull requests. When I first learned about pair programming through the sprint, I thought it was a great idea since these pairs could be arranged in such a way that participants with relatively little experience (like myself) could be paired up with more experienced developers to tackle complicated issues that they otherwise would not have been able to.

On the day of the sprint, I was eagerly waiting for my partner to come join Discord. To my disappointment, however, they did not show up. Fortunately, thanks to the support from the organizing team (shoutout to [Noemi](http://www.noemiderzsy.com)), I was quickly matched up with another participant. Together, my partner and I worked on a total of 1.5 issues for about two hours. I say 1.5 because we successfully pushed out a pull request (which has now been merged!) and another draft PR in progress that was eventually closed due to the impossibility of retrieving word columns for the `rcv1` Reuters news dataset. Another minor PR I submitted a few days prior to the sprint also got merged. Although the issues addressed were minor in significance, overall it was a highly rewarding experience. Contributing to open source has an oddly addicting nature to it---once a docs improvement PR is merged, you might find yourself scrolling through GitHub, trying to find issues relating to API improvements or feature requests that seem interesting, yet also doable by your standards.

Of course, not everything in the sprint panned out nicely for our team. For one, the sprint started at 1 in the morning in Korean Standard Time, so I wasn't sure if I would be able to stay up until 5AM. I ended up finishing a bit short a little past 4. My partner also decided to leave a bit early on their end, so thankfully things worked out well. 

Besides the inevitable (and entirely anticipated) timezone issue, however, my partner and I also struggled to set up our workflow on GitHub. My initial thought was that one of us would have to add the other as a collaborator on a forked repository. A few moments later, however, I was misguided by another idea, that perhaps we could fetch from a PR; for instance, my partner could open a PR, and I would fetch from that PR, make modification on my local, then push it up back to remote. It turned out that this was the wrong approach; pushing from my local wouldn't update the PR since tracking doesn't quite work that way. In the end, we resorted back to our initial plan of collaborating on a branch in a forked repository. Hopefully, things worked out well from there on.

Despite it being entirely virtual, the Scikit-learn sprint was organized extremely well. Discord was the main channel of communication, as well as Zoom for an introductory orientation prior to the actual sprint. Organizers and core developers provided assistance to participants with various aspects of the sprint---our team, in particular, received helpful guidance with Git as we were setting up our workflow as described earlier. They also provided prompt feedback on opened PRs, kindly pointing out which test had failed for what reason, and how one might navigate the problem.

Although I had contributed to open source [before](https://jaketae.github.io/study/development/open-source/), this was my first time working together with a partner. In retrospect, this collaborative nature is what made the sprint all the more enjoyable. It also introduced me to other helpful coding practices, such as unit testing and linting---I had been using [black](https://black.readthedocs.io/en/stable/) for auto-foramtting, but exposure to [flake8](https://flake8.pycqa.org/en/latest/) was certainly helpful. I'll also shamelessly stick in here the fact that it was simply great interacting with Scikit-learn's core developers like [Andreas Mueller](https://amueller.github.io), whom I had only seen on video lectures and ML-related articles. And of course, do I even have to mention that Scikit-learn is objectively the coolest ML package there is in Python?

Thanks again to [Data Umbrella](https://www.dataumbrella.org) and [NYC PyLadies](https://nyc.pyladies.com) for organizing this sprint!

