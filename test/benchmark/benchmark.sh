#!/bin/bash
#benchmark of images. 
#assumes all types (stimuli, ground-truth map/binary-mask have the same 
#filenames, except for possibly different extensions.

#directory where everything will be stored.
#if function gen_data is not called, assumes main_dir still has the structure
#created by gen_data.
main_dir="/home/erik/mit_300"
#directories for each type of image
gt_masks_dir="$main_dir/masks"
gt_points_dir="$main_dir/points"
gt_maps_dir="$main_dir/gt_maps"
maps_dir="$main_dir/maps"
stimuli_dir="$main_dir/stimuli"

#source original images directory
src_stimuli_dir="/home/erik/proj/ic/saliency_benchmarks/bms/cssd/images"
#source stimuli extension
stimuli_ext=".jpg"
#use ground-truth masks
use_gt_masks=true
#ground truth masks directory
src_gt_masks_dir="/home/erik/proj/ic/saliency_benchmarks/bms/cssd/ground_truth_mask"
#ground truth masks extension
gt_masks_ext=".png"
#use ground-truth points
use_gt_points=false
#ground truth points directory
src_gt_points_dir="/home/erik/proj/ic/saliency_benchmarks/bms/mit_300/BenchmarkIMAGES/"
#ground truth points extension
gt_points_ext=".jpg"
#use ground-truth maps
use_gt_maps=false
#ground truth maps directory
src_gt_maps_dir="/home/erik/proj/ic/saliency_benchmarks/bms/mit_300/BenchmarkIMAGES/"
#ground truth maps extension
gt_maps_ext=".jpg"

#number of random images to take from source images dir
sample=2

#command to run model on single image. format: $cmd <img> [flags]
att_cmd="/home/erik/proj/att/att/test.py im"
att_cmd_flags="-D -s $maps_dir -m col,cst"

#command to execute some benchmark metric in format $bm <map> [...]
bm_cmd="/home/erik/proj/att/test/benchmark/metrics.py"
bm_cmd_flags=""

#result images format
map_ext="_final_map.png"

#result file
bm_file="$main_dir/benchmark.txt"

#stats calculating script
stats_cmd="/home/erik/proj/att/test/benchmark/bm_stats.py"
stats_file="$main_dir/stats.txt"

#gets only filename of path, without extension. 
#ex: /home/foo/bar.jpg -> bar
fname()
{
	echo $(basename $1) | rev | cut -f1 -d. --complement | rev
}

#creates directory, exitting in case of any error
mkdir_check()
{
	mkdir -- $1 || { echo "midir_check: could not create $1"; exit 1; }
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
	cp -- $(ls "$src_stimuli_dir"/* | shuf | head -n $sample) "$stimuli_dir"

	#getting ground-truth masks
	if "$use_gt_masks"; then
		for f in "$stimuli_dir"/*; do
			cp -- "$src_gt_masks_dir/$(fname $f)$gt_masks_ext" "$gt_masks_dir"
		done
	fi
	#getting ground-truth points
	if "$use_gt_points"; then
		for f in "$stimuli_dir/"*; do
			cp -- "$src_gt_points_dir/$(fname $f)$gt_points_ext" "$points_dir"
		done
	fi
	#getting ground-truth maps
	if "$use_gt_maps"; then
		for f in "$stimuli_dir/"*; do
			cp -- "$src_gt_maps_dir/$(fname $f)$gt_maps_ext" "$maps_dir"
		done
	fi
}

#runs model on each source image
run()
{
	for f in "$stimuli_dir"/*; do
		#echo "in $f..."
		time $att_cmd "$f" $att_cmd_flags
		echo
	done	
}

bm()
{
	for st in "$stimuli_dir"/*; do
		map="$maps_dir/$(fname $st)$map_ext"
		gt_mask="$gt_masks_dir/$(fname $st)$gt_masks_ext"
		gt_map="$gt_maps_dir/$(fname $st)$gt_maps_ext"
		gt_points="$gt_points_dir/$(fname $st)$gt_points_ext"
		echo "in $map..."

		printf "mae: "; $bm_cmd "mae" "$map" "$gt_mask" $bm_cmd_flags
		printf "auc_judd: "; $bm_cmd "auc_judd" "$map" "$gt_mask" $bm_cmd_flags
		echo
	done
}

main()
{
	echo "generating data..."
	gen_data
	echo "done."

	echo "running model..."
	run 2>&1 | tee -a $bm_file
	echo "done."

	echo "executing benchmark..."
	bm 2>&1 | tee -a $bm_file
	exit 1

	#echo "copying generator script to benchmark dir..."
	cp -- "$0" $main_dir/gen_script.sh

	echo "getting some benchmark statistics..."
	(cd $main_dir && $stats_cmd > $stats_file)
}

main

exit 0
