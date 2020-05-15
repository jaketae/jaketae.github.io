#!/bin/sh

nb=$1

function convert(){
	jupyter nbconvert --to markdown $nb
	python edit.py ${nb%.ipynb}.md
	mv ${nb%.ipynb}.md ../_posts/
    if [[ -f ${nb%.ipynb}_files ]]
    then
        mv ${nb%.ipynb}_files ../assets/images/
    fi
	echo "==========Conversion complete!=========="
	sublime .
}

convert