#!/bin/bash
for i in "$@"; do 
    echo $i
    ext=${i##*.}
    if [[ $ext == "asc" ]]; then
        sed -i $1 -e '1,3s/ /\t/g' 
        sed -i $1 -e '4,$s/ //g'
    else
        echo "Skipping $1, extension is $ext"
    fi
done
