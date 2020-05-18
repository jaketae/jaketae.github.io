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
    elif opt == "-p":
        ipynb(nb)
    else:
        print("Please enter a valid option, one of -r or -p.")
