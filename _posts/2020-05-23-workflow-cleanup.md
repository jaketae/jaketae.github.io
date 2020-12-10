---
title: Blog Workflow Cleanup
mathjax: false
toc: true
categories:
  - blog
tags:
  - jupyter
  - r
---

These past few days, I've been writing posts on R while reading Hadley Wickham's [R for Data Science](https://r4ds.had.co.nz). R is no Python, but I'm definitely starting to see what makes R such an attractive language for data analysis. In particular, writing texts and code blocks on RStudio has been a good experience so far, keeping me focused and productive in my quest to R mastery (I'm nowhere near close). 

However, this newfound interest of mine has created a bit of a turbulence in my blog workflow. Previously, the vast majority of my work fell into one of two categories:

* A math-oriented post that only required Mathjax on markdown
* A code-oriented post written on Jupyter notebooks, mostly with Python

Therefore, the [automation workflow](https://jaketae.github.io/blog/jupyter-automation/) that I had set up using `nbconvert` worked perfectly fine, since I could simply write a Jupyter notebook, then convert it into markdown format and do make whatever minor modifications are necessary prior to publishing. However, this certainly did not work for R notebooks, or R Markdown documents, which go by the extension `.Rmd` . Although `.ipynb` and `.Rmd` files are functionally somewhat similar in that they both allow people to write both texts and execute code blocks all in one document, they are distinct file formats, and hence the `nbconvert`-based workflow had to be modified. 

# Knit

`knit` is a functionality in RStudio that allows people to export `.Rmd` files into other file formats, such as HTML, PDF, or even `.md` markdown documents. Obviously, given the current setup of my blog, which uses Jekyll, I need to export the notebooks into markdown format. So this was great news. 

## YAML Setup

Here is the setup I use for knitting my document.

```yaml
---
title: some_title
categories:
  - some_category
tags:
  - some_tag
output: 
  md_document:
    preserve_yaml: true
    toc: false
    fig_retina: 2
---
```



The part that matters most is the `output` section of the YAML front matter. Here, we specify that we want to export the R notebook as a `md_document`, which stands for markdown document. I also realized that setting `preserve_yaml` to `true` saves me a bit of hassle since the `title`, `categories`, and `tags` can be pre-configured within the R notebook. With this YAML header, clicking on the green knit button will make RStudio do its thing and churn out a `.md` file, as well as an image folder that contains the plots and graphs created from executing the code blocks in the notebook. 

## File Directories

However, the knit button is certainly not a magic button. One problem with the knit function is that, by default, it creates a directory of image files in the same location where the notebook is located. In my case, this was the `jaketae.github.io/_jupyter` directory. In other words, knitting would result in the creation of a directory like  `jaketae.github.io/_jupyter/some_image_directory`. Moreover, the knitted `.md` file would also be in the `/_jupyter` directory.

However, this was not how I had organized my files. Here is a heavily truncated summary of how the blog is currently organized. A lot of irrelevant directories containing other customizations were omitted in this summary. In case you're wondering how to create this summary, run

```bash
brew install tree
tree some_directory
```

Here is the result I got with my blog directory.

```
jaketae.github.io
├── Gemfile
├── README.md
├── _config.yml
├── _drafts
│   ├── some_draft_1.md
│   └── some_draft_2.md
├── _jupyter
│   ├── some_ipynb_notebook.ipynb
│   ├── some_r_notebook.Rmd
│   ├── format.py
│   └── format.sh
├── _posts
│   ├── some_post.md
│   └── another_post.md
├── assets
│   ├── css
│   │   └── main.scss
│   └── images
│       ├── some_image_directory
│       │   ├── some_image.png
│       │   └── another_image.png
│       └── some_image.png
└── index.md

```

As you can see, the images are located in the `/assets/images` directory. Also, the notebooks and the posts are in separate locations: while the notebooks reside in `/_jupyter`, the posts live in `/_posts`. Therefore, knitting would mean that I would have to 

* Move the knitted `.md` file to `/_posts`
* Move the knitted images to `/assets/images`

Fortunately, it's pretty easy to achieve these tasks using the linux shell. In fact, this is very similar to what we had done previously with Jupyter notebooks: only this time, R notebooks were added into the mix. Therefore, I had to update the existing shell script to account for both `.ipynb` and `.Rmd` file separately. 

# Scripting

Being lazy programmers, we always want to automate things as much as possible. My end goal was to achieve all the knitting, converting, and moving in just a single line on the command prompt. 

## Shell Scripting

The first thing I had to do was to update the Python and shell scripts that I had previously for handling `.ipynb` files. Let's first take a look at the previous shell script I had, introduced in this post.

```shell
#!/bin/sh

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

What this script does is pretty simple: it uses `nbcovert` to create a `.md` file from a `.ipynb` file. Then, it uses a Python script to make some edits. Lastly, we move the image and markdown files to appropriate locations. 

Here is the script after the revamp, with added functionality to account for `.Rmd` files as well.

```shell
#!/bin/sh

nb=$1

function ipynb(){
	jupyter nbconvert --to markdown $nb
	python format.py "-p" ${nb%.ipynb}.md
	mv ${nb%.ipynb}.md ../_posts/
    if [[ -d ${nb%.ipynb}_files ]]
    then
        echo "==========Moving image files=========="
        mv ${nb%.ipynb}_files ../assets/images/
    fi
}

function rmd(){
    eval "$(conda shell.bash hook)"
    conda activate R
    R -e "rmarkdown::render('$nb')"
    python format.py "-r" ${nb%.Rmd}.md
    mv ${nb%.Rmd}.md ../_posts/
    if [[ -d ${nb%.Rmd}_files ]]
    then
        echo "==========Moving image files=========="
        mv ${nb%.Rmd}_files ../assets/images/
    fi
}

function format(){
    if [[ $nb == *.ipynb ]]
    then
        echo "==========Starting .ipynb conversion=========="
        ipynb
    else
        echo "==========Starting .Rmd formatting=========="
        rmd
    fi
    echo "==========Formatting complete!=========="
    cd ../_posts
    open ${nb%.*}.md
}

format
```



The `format()` function can be seen as the driver program that fires different functions depending on the extension of the input file, specified as an argument on the command line. The `ipynb()` function is nearly identical to what we had before. The only change is that the function now performs a preliminary check of whether or not an image directory exists before attempting to move it. 

The addition lies in `rmd()`, which is a function that performs both the knitting, editing, and the moving. Due to the Anaconda-based R setup I've described in [this post](https://jaketae.github.io/development/anaconda/), using the `R` command in the terminal requires the activation of the relevant virtual environment. After some searching, I realized that this can be done with `    eval "$(conda shell.bash hook)"`. After activating the environment, knitting is performed via `R -e "rmarkdown::render('$nb')"`. Then, it's the same drill all over again: edit the file with a Python script, then move the files to their respective locations in the blog directory. 

When all of this is done, we open the newly created markdown file in the `/_posts` directory using our favorite markdown editor, [Typora](https://jaketae.github.io/blog/typora/).

## Python Scripting

The reason why we need a Python script is that the image hyperlinks are most often broken, leading to rendering issues. This issue was documented in the previous post for the `.ipynb` case with `nbcovert`; hence the regular expressions with find and replace you see in the `ipynb()` function. 

Knitting on R does not engender the same issue. In fact, it works without issues. The issue arises, however, when we move the image file directory to `/assets/images`. When the directory is moved, the hyperlinks start to break down. Therefore, we have to make some adjustments. This improved Python script accounts for both cases of `.ipynb` and `.Rmd` files, much like the shell script is able to handle both cases.

```python
#!/usr/bin/python

import re
import sys


def rmd(nb):
    path = f"/Users/jaketae/Documents/GitHub/jaketae.github.io/_jupyter/{nb}"
    with open(path, "r") as file:
        filedata = file.read()
    filedata = re.sub('src="', 'src="/assets/images/', filedata)
    with open(path, "w") as file:
        file.write(filedata)


def ipynb(nb):
    path = f"/Users/jaketae/Documents/GitHub/jaketae.github.io/_jupyter/{nb}"
    with open(path, "r") as file:
        filedata = file.read()
    yaml = "---\ntitle: TITLE\nmathjax: true\ntoc: true\ncategories:\n  - category\ntags:\n  - tag\n---\n\n"
    filedata = re.sub(r"!\[svg\]\(", '<img src="/assets/images/', filedata)
    filedata = re.sub(".svg\)", '.svg">', filedata)
    filedata = re.sub(r"!\[png\]\(", '<img src="/assets/images/', filedata)
    filedata = re.sub(".png\)", '.png">', filedata)
    filedata = yaml + filedata
    with open(path, "w") as file:
        file.write(filedata)


if __name__ == "__main__":
    opt = sys.argv[1]
    nb = sys.argv[2]
    if opt == "-r":
        rmd(nb)
    else:
        ipynb(nb)

```

While going about this task, I ran into a number of issues. Among them, the most important one was to configure the Python script to be able to handle `.Rmd` and `.ipynb`-based markdown files. If the knitted or converted file are both markdown files, how would the Python script know which markdown file was created from a R markdown or a IPython notebook? I decided to resolve this by passing an additional argument to the Python script; hence the `sys.argv[1]`, which is the `-r` or `-p` flag. But in the end, this flag is hidden to the user--it is a hidden API--and thus does not introduce additional complexities to the usability of the script.

# Conclusion

Shell scripting turned out to be more interesting than I had thought. I'm satisfied because a lot of manual work is now the computer's job; as a writer, I can now focus only on publishing quality content without having to fiddle with files or minor formatting issues with hyperlinks to images. This works with both IPython and R notebooks, which was the end goal of this little endeavor. 

Shell scripting is something I hope to continue doing. If there is one thing I learned, it is that Python + Shell is an incredibly powerful combination that never goes wrong.