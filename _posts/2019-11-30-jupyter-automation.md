---
title: Using Jupyter Notebook with Jekyll
date: 2019-11-30
categories:
  - blog
tags:
  - jupyter
---

In the [last post], I tested out the functionality of [Jupyter Notebook], a platform that I am just starting to get acquainted with. I'm pleased with how that experimental post turned out, although there are still things to modify and improve. In this post, I explain how I was able to automate the process of converting Jupyter Notebook into markdown using shell script and python commands.

# Motivation

This blog runs on [GitHub pages] using [Jekyll], a static website generator engine. When I conceived of the idea of starting a personal blog about a month ago, I considered various hosting options, such as [Wordpress] or [Wix], but I eventually settled on Jekyll, convinced by its utility as well as customizability. I also considered other options such as [Hugo] or [Gatsby], but decided that Jekyll was more starter-friendly and offered to do things right out of the box. 

Jekyll makes use of `.md` files to display posts. For the most part, uploading posts is a painless process as markdown files are very easy to create. However, one drawback of using this file format with Jekyll is that embedding images in posts requires me to download and place images in the correct path, namely `assets/images` in the GitHub repository. Because most of the images I use are graphs and plots generated from executing blocks of code on Python, this manual reference protocol requires that I follow a rather roundabout routine:

* Write python code on Sublime or PyCharm
* Call the `plt.savefig()` method to save generated figures on local drive
* Move image file to `assets/images` directory
* Copy and paste relevant code manually onto the `.md` file
* Delete the `plt.savefig()` line
* Reference image file in `assets/images/` in `.md` file

While this is not a complicated process, it is a time-consuming one indeed, especially if the post gets longer. Another downside of this workflow is that it is prone to human error because most of the steps are performed manually by me. Copy and pasting, writing links, and moving files across different directories are all tasks that require human input. For those reasons, I found the my workflow to be inefficient. Now enters automation and Jupyter Notebooks into the picture. 

# Jupyter Notebook

The biggest advantage of using Jupyter Notebooks is that plots and figures are generated on the document directly. In other words, there is no need to write code in one location and copy and paste it to another, neither do I have to save a figure as an image file and link it in the post separately---code, figures, and texts all stay nicely in one package. After all, the whole idea behind Jupyter Notebook is to present information in a readable, accessible fashion. This is why I found Jupyter Notebook to be a very convenient option: it streamlined the process of generating figures and writing explanations for my blog posts, specifically into the following three steps:

* Write Jupyter Notebook
* Run each code block
* Save file by generating a checkpoint

And now I'm done! Notice how the complicated, inefficient workflow above has shrunk into three, nice and simple steps. The Jupyter Notebook UI is also highly intuitive, which is why it did not take a lot of time for me to start writing code and generating figures after fidgeting around with some options for a bit. 

However, Jupyter Notebook is not without its problems. For one, I had to find a method to integrate Jupyter Notebooks into Jekyll. Jupyter Notebook comes in `.ipynb` file format, which is not presentable on GitHub pages by default. This meant that I had to convert Jupyter Notebook into something like `.html` or `.md`, the two most commonly used file types in Jekyll. Fortunately, I found a command line conversion tool, named `nbconverter` that performed this task quite painlessly. After some experimentation, I was able to generate a `.md` file of the Jupyter Notebook. 

So I should be done, right?

Well, unfortunately not. Examining the end product, I realized two glaring problems. First, the converted file did not include any [YAML front matter]. Basically, all posts on Jekyll include some header that contains information on their title, tag, category, and date information. The converted file, however, did not include this header because it was converted from Jupyter Notebooks which inherently does not have the notion of YAML front matter. This meant that I had to manually type in these information again, which is, to be fair, not such a time consuming process, but nonetheless additional work on my part. 

A more serious issue with this conversion was that all the image links were broken. This was certainly bad news: when I compiled and built the website on GitHub, I found that none of the images were properly displayed on the web page. Upon examination, I found that all the dysfunctional links took the following form:

```
![png](some_file_name.png)
```

Proper image links, on the other hand, look like

```
<img src="/assets/images/some_file_name.png">
```

On a quick side note, those of you who know `.html` will easily recognize that this conforms to `.html` syntax. Anyhow, to make this work, I realized that I would have to walk through every image link in the converted file and restructure them correctly into standard `.html` tag format. This sounded intimidating, but I decided to try it nonetheless. After performing this menial task for some time, however, I concluded that this manual handwork was not the way to go. 

# Automating with Python

This is where the fun part comes in. I decided to implement a simple automation system that would solve all of my problems and worries. Specifically, this script would have to perform the following list of tasks:

* Convert Jupyter Notebook to markdown
* Add YAML front matter to the converted file
* Fix broken image links
* Move the converted file to `_posts` directory
* Move image files to `/assets/images/` directory

This might seem a lot to handle, but some simple scripting can do the trick. First, we begin by creating a `.sh` script that will form the basis of this automating process. I used the hash bang `#!/bin/sh` to make this an executable. The script itself is very short---it goes as follows.

```
nb=$1

function convert(){
	jupyter nbconvert --to markdown $nb
	python edit.py ${nb%.ipynb}.md
	mv ${nb%.ipynb}.md ../_posts/
	mv ${nb%.ipynb}_files ../assets/images/
	echo "==========Conversion complete!=========="
}

convert
```

The first line of the function invokes a `nbconvert` command, the result of which is the creation of a `.md` converted file. Then, the script runs another a Python script, but I will get more into that in a second. After Python does its portion of the work, I use two `mv` commands to move the `.md` file to the `_posts` directory and the associated image files to the `/assets/images/` directory, as specified above. Then, the script echoes a short message to tell the user that the process is complete. Pretty simple, yet it does its trick quite nicely. 

Now onto the Python part. If you take a look back at the task list delineated above, you will see that there are two bullet point that has not been ticked off by the `convert.sh` script: adding YAML front matter and fixing broken image links---and that is exactly what our `edit.py` script will do! Let's take a look at the code:

```python
#!/usr/bin/python

import sys, re

def edit():
	path = "/Users/jaketae/Documents/GitHub/jaketae.github.io/_jupyter/" + str(sys.argv[1])
	yaml = "---\ntitle: TITLE\nmathjax: true\ncategories:\n  - category\ntags:\n  - tag\n---\n\n"
	with open(path, 'r') as file:
		filedata = file.read()
	filedata = re.sub(r"!\[png\]\(", "<img src=\"/assets/images/", filedata)
	filedata = re.sub(".png\)", ".png\">", filedata)
	filedata = yaml + filedata
	with open(path, 'w') as file:
		file.write(filedata)

if __name__ == '__main__':
	edit()
```

In a nutshell, the script opens and reads the converted `.md` file, finds and replaces broken image links with correct ones, and adds the default YAML front matter, defined as the string `yaml` in the code above. When all the necessary edits are performed, the script then rewrites the `.md` file with the edited content. 

This script works fine, but I feel like all of this might have been implemented a lot more efficiently if I were proficient with regular expressions. The task of locating and editing broken image links into `html` tags is a classic `regexp` problem, but my knowledge of Python and regular expressions were not quite enough to implement the most efficient mechanism to tackle this problem. Nonetheless, my solution still produces the desired result, which is why I am very content with my first attempt at automation with Python and script. 

# Conclusion

Now, I have a completely revised workflow that suits my purposes the best: Jupyter Notebooks to write code and generate figures alongside text explanations, and an automation system that converts these Jupyter Notebooks to markdown efficiently. All I have to do is to edit the default YAML front matter, such as coming up with good titles and categorizing my post to appropriate categories. 

This automation script is by no means perfect, which is why I plan on incremental updates as I go. Specifically, I might add additional features, such as the ability to auto-generate figure captions. I might also streamline the process of find and replace that occurs within the Python script after learning more about regular expressions. But for now, the project seems solid as it is, and I am happy with what I have. 

Having completed this little script, I am thinking of starting another automation project that will help me with my work at the PMO. This is going to be a much tougher challenge that involves reading in `.pdf` files and outputting `.xlsx` files, but hopefully I can streamline my workflow at the office with Python's help. I am excited to see where this project will go, and I will definitely keep this posted. In the meanwhile, happy thanksgiving!


[last post]: https://jaketae.github.io/blog/test/
[Jupyter Notebook]: https://jupyter.org
[Wordpress]: https://wordpress.com
[Wix]: https://www.wix.com
[Hugo]: https://gohugo.io
[Gatsby]: https://www.gatsbyjs.org
[YAML front matter]: https://jekyllrb.com/docs/front-matter/
[Jekyll]: https://jekyllrb.com
[GitHub pages]: https://pages.github.com
