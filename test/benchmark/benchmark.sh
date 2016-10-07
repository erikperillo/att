#!/bin/bash
#benchmark of images. assumes they all have the same filenames.

#source images directory
#bm_imgs_dir=~/proj/ic/saliency_benchmarks/bms/cssd/images
#bm_imgs_dir=~/proj/ic/saliency_benchmarks/bms/ecssd/images
bm_imgs_dir=~/proj/ic/saliency_benchmarks/bms/mit_300/BenchmarkIMAGES/BenchmarkIMAGES/SM
#bm_imgs_dir=~/proj/att/test/imgs
#source images extension
bm_imgs_ext=".jpg"
#benchmarks mask images directory
bm_masks_dir=~/proj/ic/saliency_benchmarks/ecssd/ground_truth_mask
#mask images extension
bm_masks_ext=".png"
#command to execute benchmark in format: $cmd <bm_img> <img>
bm_cmd="./test.py bm"
#number of random images to take from source images dir
sample=300
#directory where everything will be stored.
#if gen_data is not called, assumes main_dir still has the structure
#created by gen_data.
main_dir=~/mit_300
#command to run model on single image. format: $cmd <img> [flags]
att_cmd_dir=/home/erik/proj/att/att
att_cmd="./test.py im"
att_cmd_flags="-D -s $main_dir/results -m col,cst"
#command to run benchmark on pair of benchmark mask and image. 
#format: $cmd <mask_bm_img> <img> [flags] <mask_img>
bm_cmd_dir=/home/erik/proj/att/att
bm_cmd="./test.py bm"
bm_cmd_flags="-D -M -m"
#result images format
#map_ext="_final_map.png"
map_ext="_final_map.png"
map_mask_ext="_final_map_mask.png"
#result file
bm_file=$main_dir/benchmark.txt
#stats calculating script
stats_cmd=/home/erik/proj/att/test/benchmark/bm_stats.py
stats_file=$main_dir/stats.txt

fname()
{
	echo $(basename $1) | rev | cut -f1 -d. --complement | rev
}

mkdir_check()
{
	mkdir -- $1 || { echo "could not create $1"; exit 1; }
}

gen_data()
{
	mkdir_check $main_dir
	mkdir_check $main_dir/imgs
	mkdir_check $main_dir/masks
	mkdir_check $main_dir/results

	#getting images
	cd $bm_imgs_dir
	cp -- $(ls | shuf | head -n $sample) $main_dir/imgs
	cd - > /dev/null

	#getting masks
	for f in $main_dir/imgs/*; do
		cp -- $bm_masks_dir/$(fname $f)$bm_masks_ext $main_dir/masks
	done
}

run()
{
	cd $att_cmd_dir

	for f in $main_dir/imgs/*; do
		#echo "in $f..."
		time $att_cmd $f $att_cmd_flags
		echo
	done	
}

bm()
{
	cd $bm_cmd_dir

	for bm_f in $main_dir/masks/*; do
		f=$main_dir/results/$(fname $bm_f)$map_ext
		mask_f=$main_dir/results/$(fname $bm_f)$map_mask_ext
		if [ -f "$f" ]; then
			echo "comparing $bm_f and $f"
			$bm_cmd $bm_f $f $bm_cmd_flags $mask_f
			echo
		fi
	done
}

main()
{
	#echo "generating data..."
	gen_data

	#echo "running model..."
	run 2>&1 | tee -a $bm_file

	echo "executing benchmark..."
	bm 2>&1 | tee -a $bm_file

	#echo "copying generator script to benchmark dir..."
	cp -- "$0" $main_dir/gen_script.sh

	echo "getting some benchmark statistics..."
	(cd $main_dir && $stats_cmd > $stats_file)
}

main

exit 0
