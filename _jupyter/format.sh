#!/bin/sh

nb=$1

function format(){
    mv ${nb%.md}_files ../assets/images/
    python format.py $nb
    mv $nb ../_posts/
    echo "==========Formatting complete!=========="
}

format