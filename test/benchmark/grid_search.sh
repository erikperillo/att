#!/bin/bash

pyr_lvls="2 3 4"
cc_scores="num cmdrssq cmdrsqm"
col_ws="0.5 1.0 2.0"
cst_ws="0.5 1.0 2.0"

base_dir="/home/erik/grid_search"
combs=0

sec_to_min()
{
	python -c "print($1/60)"
}

for pl in $pyr_lvls; do
	for ccs in $cc_scores; do
		for colw in $col_ws; do
			for cstw in $cst_ws; do
				flags="-m col,cst -p $pl -n $ccs --colw $colw --cstw $cstw"	
				dir=$base_dir/"col_cst_p-$pl""_n-$ccs""_colw-$colw""_cstw-$cstw"
				echo "flags = $flags"
				echo "dir = $dir"	
				echo "executing..."
				#sleep 1
				start_t=$(date +%s)	
				./benchmark.sh "$dir" "$flags"
				end_t=$(date +%s)	
				echo "time elapsed: $(sec_to_min $((end_t - start_t))) mins"
				#sleep 1
				echo
				combs=$((combs + 1))
			done
		done
	done
done

echo "combs = $combs"
