#!/usr/bin/python

import sys, re

def edit():
	path = "/Users/jaketae/Documents/GitHub/jaketae.github.io/_jupyter/" + str(sys.argv[1])
	with open(path, 'r') as file:
		filedata = file.read()
	filedata = re.sub(r"!\[png\]\(", "<img src=\"/assets/images/", filedata)
	filedata = re.sub(".png\)", ".png\">", filedata)
	with open(path, 'w') as file:
		file.write(filedata)

if __name__ == '__main__':
	edit()