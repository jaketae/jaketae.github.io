#!/bin/sh

nb=$1

function convert(){
	jupyter nbconvert --to markdown $nb
	mv ${nb%.ipynb}.md ../_posts/
	mv ${nb%.ipynb}_files ../assets/images/
	echo "Conversion complete!"
}

convert