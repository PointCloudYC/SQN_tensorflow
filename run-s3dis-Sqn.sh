#!/bin/bash

MODES=('train')
# MODES=('train' 'test')
gpu=0
# 4 costs about 8GB GPU, others can be used on grp server GPUs
BATCH_SIZES=(4) # 16 10 8 6 
# VAL_BATCH_SIZES=(20) # 16 10 8 6 
# 8 works on my nvidia 1080 gpu, others can be tried on grp server GPUs
VAL_BATCH_SIZES=(8) 
num_points=40960
max_epoch=400
# WEAK_LABEL_RATIOS=(0.1 0.01 0.001 0.0001)
WEAK_LABEL_RATIOS=(0.001 0.0001)
# WEAK_LABEL_RATIOS=(0.001)

# TODO: ablation study
# num_k_query_pts
# how to concat features
# CONCAT_TYPES=('1234' '123' '234' '12' '1')
CONCAT_TYPES=('1234')

for mode in "${MODES[@]}"; do
	for weak_label_ratio in "${WEAK_LABEL_RATIOS[@]}"; do
			for batch_size in "${BATCH_SIZES[@]}"; do
				for val_batch_size in "${VAL_BATCH_SIZES[@]}"; do
                         for concat_type in "${CONCAT_TYPES[@]}"; do
                              echo "batch_size: ${batch_size}"
                              echo "val_batch_size: ${val_batch_size}"
                              echo "num_points: ${num_points}"
                              echo "max_epoch: ${max_epoch}"
                              echo "weak_label_ratio: ${weak_label_ratio}"
                              echo "concat_type: ${concat_type}"

                              time python -B main_S3DIS_Sqn.py \
                              --gpu ${gpu} \
                              --mode ${mode} \
                              --test_area 5 \
                              --batch_size ${batch_size} \
                              --val_batch_size ${val_batch_size} \
                              --num_points ${num_points} \
                              --max_epoch ${max_epoch} \
                              --weak_label_ratio ${weak_label_ratio} \
                              --concat_type ${concat_type}

                              echo "batch_size: ${batch_size}"
                              echo "val_batch_size: ${val_batch_size}"
                              echo "num_points: ${num_points}"
                              echo "max_epoch: ${max_epoch}"
                              echo "weak_label_ratio: ${weak_label_ratio}"
                              echo "concat_type: ${concat_type}"
                    done
			done
		done
	done
done

echo "finish training."