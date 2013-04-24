#!/bin/bash
function usage {
    echo "Usage: $0 <remote username> <remote password> <remote ip/dns> <remote dir> <local dir>"
    echo "Will run superdiff on all matching files in remote and local directory"
    echo "Assumes superdiff is on the path or in the local dir"
}

if [[ $# -lt 5 ]]; then
    usage
    exit 1
fi

RUSER=$1
RPASS=$2
RNAME=$3
RDIR=${4%/}
LDIR=${5%/}
#Secondary directory
SDIR=$6
for f in $(ssh $RUSER@$RNAME "ls $RDIR"); do 
    LPATH=$LDIR/$f
    RPATH=$RDIR/$f
    if [[ ! -f "$LPATH" ]]; then
        if [[ ! -z $SDIR ]]; then
            LPATH=$SDIR/$f
            if [[ ! -f "$LPATH" ]]; then
                #echo "ERROR: $f does not exist in $LDIR or $SDIR!"
                continue
            fi
        else
                #echo "ERROR: $f does not exist in $LDIR!" 
                continue
        fi
    fi
    bash superdiff.sh $RUSER?$RPASS@$RNAME:$RPATH $LPATH 
done

