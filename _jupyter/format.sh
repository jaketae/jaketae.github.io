#!/bin/sh


opt=$1 # -r for .Rmd or -p for .ipynb
nb=$2


function ipynb(){
	jupyter nbconvert --to markdown $nb
	python format.py $opt ${nb%.ipynb}.md
	mv ${nb%.ipynb}.md ../_posts/
    if [[ -d ${nb%.ipynb}_files ]]
    then
        echo "==========Moving image files=========="
        mv ${nb%.ipynb}_files ../assets/images/
    fi
}


function rmd(){
    python format.py $opt $nb
    mv $nb ../_posts/
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
}


format