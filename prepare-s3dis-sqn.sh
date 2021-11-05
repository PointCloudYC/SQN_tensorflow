#!/bin/bash

# SUB_GRID_SIZES=(0.02)
SUB_GRID_SIZES=(0.04)
# WEAK_LABEL_RATIOS=(0.00001 0.00002 0.00005) 
# WEAK_LABEL_RATIOS=(0.00001 0.00002 0.00005) 
# WEAK_LABEL_RATIOS=(0.005) 
# WEAK_LABEL_RATIOS=(0.0006)
WEAK_LABEL_RATIOS=(0.001 0.0001 0.0005 0.0002)
# WEAK_LABEL_RATIOS=(0.0005 0.0002 0.0001)
# WEAK_LABEL_RATIOS=(0.1 0.01 0.0001)
# WEAK_LABEL_RATIOS=(0.1)

for sub_grid_size in "${SUB_GRID_SIZES[@]}"; do
	for weak_label_ratio in "${WEAK_LABEL_RATIOS[@]}"; do

		echo "sub_grid_size is ${sub_grid_size}"
		echo "weak_label_ratio is ${weak_label_ratio}"

		time python utils/data_prepare_s3dis_sqn.py --sub_grid_size ${sub_grid_size} \
		--weak_label_ratio ${weak_label_ratio}

		echo "sub_grid_size is ${sub_grid_size}"
		echo "weak_label_ratio is ${weak_label_ratio}"
	done
done

echo "finish preparing the S3DIS dataset for weakly semantic segmentation!"