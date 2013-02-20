#!/bin/bash
for i in accel waterjacket Mic; do
    today_date=`date --date="today" | awk '{print $2$3}'`
    convert *$i*.png kurtosis_results_"$i"_"$today_date".pdf
done
