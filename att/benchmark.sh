#!/bin/bash
#benchmark of images. assumes they all have the same filenames.

bm_imgs_dir=~/proj/ic/saliency_benchmarks/cssd/images
bm_imgs_ext=".jpg"
bm_masks_dir=~/proj/ic/saliency_benchmarks/cssd/ground_truth_mask
bm_masks_ext=".png"
imgs_dir=~/test
imgs_ext=".png"
#in format: $cmd <bm_img> <img>
bm_cmd="./test.py bm"
sample=120

fname()
{
	echo $(basename $1) | rev | cut -f1 -d. --complement | rev
}

gen_data()
{
	mkdir ./imgs
	img_dir=$(pwd)/imgs

	#getting images
	cd $bm_imgs_dir
	for f in $(ls | shuf | head -n $sample); do
		cp $f $img_dir
	done

	cd -

	#getting masks
	mkdir ./masks
	for f in ./imgs/*; do
		cp $bm_masks_dir/$(fname $f)$bm_masks_ext ./masks
	done

	#making results dir
	mkdir ./results
}

gen_data
exit 1

for img in $imgs_dir/*; do
	bm_img=$bm_masks_dir/$(basename $img)	
	$bm_cmd $bm_img $img
done
