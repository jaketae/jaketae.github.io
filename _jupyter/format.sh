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