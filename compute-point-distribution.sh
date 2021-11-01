#!/bin/bash

SUB_GRID_SIZES=(0.04)
WEAK_LABEL_RATIOS=(0.001 0.0001)

for sub_grid_size in "${SUB_GRID_SIZES[@]}"; do
	for weak_label_ratio in "${WEAK_LABEL_RATIOS[@]}"; do

		echo "sub_grid_size is ${sub_grid_size}"
		echo "weak_label_ratio is ${weak_label_ratio}"

		time python utils/compute_weak_point_distribution.py --sub_grid_size ${sub_grid_size} \
		--weak_label_ratio ${weak_label_ratio}

		echo "sub_grid_size is ${sub_grid_size}"
		echo "weak_label_ratio is ${weak_label_ratio}"

	done
done

echo "finish computing!"