#!/bin/bash
#To regularize the filenames, use this command
#find -name "* *" -type f | rename 's/ /_/g'
for i in "$@"; do 
    echo $i
    ext=${i##*.}
    match="dat"
    if [[ $ext == $match ]]; then
        echo "Regularizing $i"
        sed -i $i -e '1,8s/ /\t/g' 
        sed -i $i -e '9s/^\t\s/Count/'
        sed -i $i -e '8,$s/ //g'
    else
        echo "Skipping $i, extension is $ext, only formatting $match"
    fi
done
