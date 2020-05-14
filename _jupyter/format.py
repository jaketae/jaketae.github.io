#!/usr/bin/python

import sys
import re


def format():
    path = f"/Users/jaketae/Documents/GitHub/jaketae.github.io/_jupyter/{sys.argv[1]}"
    with open(path, 'r') as file:
        filedata = file.read()
    filedata = re.sub("src=\"", "src=\"/assets/images/", filedata)
    with open(path, 'w') as file:
        file.write(filedata)


if __name__ == '__main__':
    format()
