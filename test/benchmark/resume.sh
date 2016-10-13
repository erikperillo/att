#!/bin/bash

#resume.sh -- resumes a grid search run by metric

dir="."
[[ ! -z "$1" ]] && dir="$1"
metric="fp_auc_judd"
[[ ! -z "$2" ]] && metric="$2"
stats_file="stats.csv"

for d in "$dir"/*; do
    #cutting for filename, metric_mean, metric_ci_min, metric_ci_max
    if [[ -d "$d" ]]; then
        echo "$d,$(cat $d/$stats_file | grep $metric)" | cut -f1,3,4,5 -d,
    fi
done
