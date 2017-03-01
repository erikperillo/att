#!/bin/bash
#benchmark of images.
#assumes all types (stimuli, ground-truth map/binary-mask have the same
#filenames, except for possibly different extensions.

#generate random data by calling gen_data
do_gen_data=true
#run model on pictures by calling run
do_run=true
#measure metrics by calling bm
do_bm=true
#compute metrics stats by calling stats
do_stats=true

#directory where everything will be stored.
#if function gen_data is not called, assumes main_dir still has the structure
#created by gen_data.
#can be changed via command line.
main_dir="/home/erik/test"
[[ ! -z "$1" ]] && main_dir="$1"
#directories for each type of image
gt_masks_dir="$main_dir/gt_masks"
gt_points_dir="$main_dir/gt_points"
gt_maps_dir="$main_dir/gt_maps"
maps_dir="$main_dir/maps"
stimuli_dir="$main_dir/stimuli"

#source original images directory
#src_stimuli_dir="/home/erik/test/stimuli"
src_stimuli_dir="/home/erik/proj/ic/saliency_benchmarks/bms/judd/stimuli"
#src_stimuli_dir="/home/erik/grid_search/judd_db/stimuli"
#source stimuli extension
stimuli_ext=".jpeg"

#result images format
map_ext="_final_map.png"

#use ground-truth masks
use_gt_masks=false
#ground truth masks directory
src_gt_masks_dir=""
#ground truth masks extension
gt_masks_ext=""

#use ground-truth fixation points
use_gt_points=true
#ground truth points directory
#src_gt_points_dir="/home/erik/test/gt_points"
src_gt_points_dir="/home/erik/proj/ic/saliency_benchmarks/bms/judd/points"
#src_gt_points_dir="/home/erik/grid_search/judd_db/gt_points"
#ground truth points extension
gt_points_ext="_fixPts.jpg"

#use ground-truth maps
use_gt_maps=true
#ground truth maps directory
#src_gt_maps_dir="/home/erik/test/gt_maps"
src_gt_maps_dir="/home/erik/proj/ic/saliency_benchmarks/bms/judd/maps"
#src_gt_maps_dir="/home/erik/grid_search/judd_db/gt_maps"
#ground truth maps extension
gt_maps_ext="_fixMap.jpg"

#number of random images to take from source images dir
n_samples=128

#command to run model on single image. format: $cmd <img> [flags]
att_cmd="/home/erik/proj/att/att/test.py im"
#flags to use. can be extended via command line.
att_cmd_flags="-D -d -s $maps_dir"
[[ ! -z "$2" ]] && att_cmd_flags="$att_cmd_flags ""$2"

#command to execute some benchmark metric in format $bm <map> [...]
bm_cmd="/home/erik/proj/att/test/benchmark/metrics.py"
bm_cmd_flags=""

#command to get statistics from metrics
stats_cmd="/home/erik/proj/att/test/benchmark/bm_stats.py"
stats_cmd_flags=""

#run function results file
run_file="$main_dir/run.txt"
#bm function results file
bm_file="$main_dir/bm.csv"
#statistics file
stats_file="$main_dir/stats.csv"
#error and log messages file
log_file="$main_dir/bm.log"

#metrics to use.
#bm_* are metrics for binary masks.
#cm_* are metrics for continuous maps.
#fp_* are metrics for fixation point maps.
bm_metrics="" #"auc_judd mae"
cm_metrics="sim cc"
fp_metrics="auc_judd nss" #"auc_judd auc_shuffled nss"

#gets random file from directory
rand_file()
{
    ls "$1" | shuf | head -n 1
}

#gets only filename of path, without extension.
#ex: /home/foo/bar.jpg -> bar
fname()
{
    echo $(basename $1) | rev | cut -f1 -d. --complement | rev
}

#creates directory, exitting in case of any error
mkdir_check()
{
    mkdir -p -- $1 || { echo "midir_check: could not create $1"; exit 1; }
}

#joins string $1 with delimiter $2.
#ex: join "a b skdj jd" "," -> "a,b,skdj,jd"
join()
{
    python -c "print('$2'.join('$1'.split()))"
}

#generates data
gen_data()
{
    #creating directories
    mkdir_check "$main_dir"
    for d in "$gt_masks_dir" "$gt_points_dir" "$gt_maps_dir" "$maps_dir" \
        "$stimuli_dir"; do
        mkdir_check "$d"
    done

    #getting images
    cp -- $(ls "$src_stimuli_dir"/* | shuf | head -n $n_samples) "$stimuli_dir"

    #getting ground-truth masks
    if "$use_gt_masks"; then
        for f in "$stimuli_dir"/*; do
            cp -- "$src_gt_masks_dir/$(fname $f)$gt_masks_ext" "$gt_masks_dir"
        done
    fi
    #getting ground-truth points
    if "$use_gt_points"; then
        for f in "$stimuli_dir/"*; do
            cp -- "$src_gt_points_dir/$(fname $f)$gt_points_ext" \
                "$gt_points_dir"
        done
    fi
    #getting ground-truth maps
    if "$use_gt_maps"; then
        for f in "$stimuli_dir/"*; do
            cp -- "$src_gt_maps_dir/$(fname $f)$gt_maps_ext" "$gt_maps_dir"
        done
    fi
}

#runs model on each source image
run()
{
    for f in "$stimuli_dir"/*; do
        #echo "in $f..."
        echo "running '$att_cmd $f $att_cmd_flags' ..."
        time $att_cmd "$f" $att_cmd_flags
        echo
    done
}

#gets benchmark
bm()
{
    #header for csv-formatted benchmark results
    [[ ! -z $cm_metrics ]] && cm_m=$(join "cm_$cm_metrics" " cm_")
    [[ ! -z $fp_metrics ]] && fp_m=$(join "fp_$fp_metrics" " fp_")
    [[ ! -z $bm_metrics ]] && bm_m=$(join "bm_$bm_metrics" " bm_")
    join "img_path $fp_m $bm_m $cm_m" ","

    for st in "$stimuli_dir"/*; do
        map="$maps_dir/$(fname $st)$map_ext"
        gt_mask="$gt_masks_dir/$(fname $st)$gt_masks_ext"
        gt_map="$gt_maps_dir/$(fname $st)$gt_maps_ext"
        gt_points="$gt_points_dir/$(fname $st)$gt_points_ext"
        other_gt_points=$gt_points
        while [ "$other_gt_points" = "$gt_points" ]; do
            rand_map=$(rand_file "$stimuli_dir")
            other_gt_points="$gt_points_dir/$(fname $rand_map)$gt_points_ext"
            if [ "$n_samples" -lt 2 ]; then
                break;
            fi
        done

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

        #binary mask metrics
        for m in $bm_metrics; do
            #printf "\tbm_$m: "
            line+=$($bm_cmd "$m" "$map" "$gt_mask" $bm_cmd_flags)" "
        done

        #continuous map metrics
        for m in $cm_metrics; do
            #printf "\tcm_$m: "
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
    if $do_gen_data; then
        echo "generating data..."
        gen_data
        echo -e "done.\n"
    fi

    if $do_run; then
        echo "running model..."
        run 2>&1 | tee "$run_file"
        echo -e "done.\n"
    fi

    if $do_bm; then
        echo "executing benchmark..."
        bm | tee "$bm_file"
        echo -e "done.\n"
    fi

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
