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
for f in $(ssh $RUSER@$RNAME "ls /usr/hfw/etc/"); do 
    bash -x superdiff.sh $RUSER?$RPASS@$RNAME:$RDIR/$f $LDIR/$f 
done

