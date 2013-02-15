#!/bin/bash
for i in "$@"; do 
    echo $i
    ext=${i##*.}
    match="asc"
    if [[ $ext == $match ]]; then
        echo "Regularizing $1"
        sed -i $i -e '1,3s/ /\t/g' 
        sed -i $i -e '4,$s/ //g'
    else
        echo "Skipping $i, extension is $ext, only formatting $match"
    fi
done
