#!/bin/sh

today=`date '+%Y-%m-%d'`
nohup ./run.sh > ./tmp/out_${today}.log 2> ./tmp/err_${today}.log < /dev/null &
