#!/bin/bash
#benchmark of images.
#assumes all types (stimuli, ground-truth map/binary-mask have the same
#filenames, except for possibly different extensions.

#compute metrics stats by calling stats
do_stats=true

#directory of predicted continuous saliency maps.
#can be changed via command line.
maps_dir="/home/erik/test"
[[ ! -z "$1" ]] && maps_dir="$1"
main_dir=$maps_dir

#result images format
map_ext=".jpg"

#use ground-truth fixation points
use_gt_points=true
#ground truth points directory
gt_points_dir="/home/erik/proj/ic/saliency_datasets/judd/points"
#ground truth points extension
gt_points_ext="_fixPts.jpg"

#use ground-truth maps
use_gt_maps=true
#ground truth maps directory
gt_maps_dir="/home/erik/proj/ic/saliency_datasets/judd/maps"
#ground truth maps extension
gt_maps_ext="_fixMap.jpg"

#command to execute some benchmark metric in format $bm <map> [...]
bm_cmd="/home/erik/proj/att/test/benchmark/metrics.py"
bm_cmd_flags=""

#command to get statistics from metrics
stats_cmd="/home/erik/proj/att/test/benchmark/bm_stats.py"
stats_cmd_flags=""

#bm function results file
bm_file="$main_dir/bm.csv"
#statistics file
stats_file="$main_dir/stats.csv"
#error and log messages file
log_file="$main_dir/bm.log"

#metrics to use.
#cm_* are metrics for continuous maps.
#fp_* are metrics for fixation point maps.
fp_metrics="auc_judd nss" #"auc_judd auc_shuffled nss"
cm_metrics="sim cc"

#gets only filename of path, without extension.
#ex: /home/foo/bar.jpg -> bar
fname()
{
    echo $(basename $1) | rev | cut -f1 -d. --complement | rev
}

#joins string $1 with delimiter $2.
#ex: join "a b skdj jd" "," -> "a,b,skdj,jd"
join()
{
    python -c "print('$2'.join('$1'.split()))"
}

#gets benchmark
bm()
{
    #header for csv-formatted benchmark results
    [[ ! -z $cm_metrics ]] && cm_m=$(join "cm_$cm_metrics" " cm_")
    [[ ! -z $fp_metrics ]] && fp_m=$(join "fp_$fp_metrics" " fp_")
    join "img_path $fp_m $cm_m" ","

    for map in $(find "$maps_dir" -name "*$map_ext"); do
        gt_map="$gt_maps_dir/$(fname $map)$gt_maps_ext"
        gt_points="$gt_points_dir/$(fname $map)$gt_points_ext"

        #echo "in $map..."
        line="$map "

        #fixation points metrics
        for m in $fp_metrics; do
            #printf "\tfp_$m: "
            if [[ "$m" == "auc_shuffled" ]]; then
                line+=$($bm_cmd "$m" "$map" "$gt_points" "$other_gt_points" \
                    $bm_cmd_flags)" "
            else
                line+=$($bm_cmd "$m" "$map" "$gt_points" $bm_cmd_flags)" "
            fi
        done

        #continuous map metrics
        for m in $cm_metrics; do
            #printf "\tcm_$m: "
            #echo $bm_cmd "$m" "$map" "$gt_map" $bm_cmd_flags
            #$bm_cmd "$m" "$map" "$gt_map" $bm_cmd_flags
            line+=$($bm_cmd "$m" "$map" "$gt_map" $bm_cmd_flags)" "
        done

        join "$line" ","
        #echo
    done
}

#gets some benchmark statistics
stats()
{
    "$stats_cmd" "$bm_file" "$stats_cmd_flags"
}

do_all()
{
    echo "executing benchmark..."
    bm | tee "$bm_file"
    echo -e "done.\n"

    if $do_stats; then
        echo "getting some benchmark statistics..."
        stats | tee "$stats_file"
        echo -e "done.\n"
    fi

    echo "copying generator script to benchmark dir..."
    cp -- "$0" $main_dir/gen_script.sh
    echo -e "done.\n"

    exit 0
}

main()
{
    temp_log=$(mktemp)
    do_all 2> "$temp_log"
    mv "$temp_log" "$log_file"
}

main
