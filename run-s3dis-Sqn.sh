#!/bin/bash

MODES=('train')
# MODES=('train' 'test')
gpu=0
# 4 costs about 8GB GPU, others can be used on grp server GPUs
BATCH_SIZES=(3) # 3 just 8GB,  4--> OOM on 1080 GPU
# VAL_BATCH_SIZES=(20) # 16 10 8 6 
# 8 works on my nvidia 1080 gpu, others can be tried on grp server GPUs
VAL_BATCH_SIZES=(4) # 6-->system collapse
num_points=40960
max_epoch=100 # 400

# KEY: weak label ratio is defined as the number of weak points over the raw poinits
# WEAK_LABEL_RATIOS=(0.001)
# WEAK_LABEL_RATIOS=(0.1 0.01 0.001 0.0001)
# WEAK_LABEL_RATIOS=(0.001 0.0001)
# WEAK_LABEL_RATIOS=(0.00001 0.00005) # 0.01 can not on 1080gpu, suffer OOM
# WEAK_LABEL_RATIOS=(0.00001 0.00002 0.00005) 
WEAK_LABEL_RATIOS=(0.0001 0.00005 0.00002) 

# 0.005 batch_size=2, OOM
# 0.001 batch_size 3, val_batch_size 4
# WEAK_LABEL_RATIOS=(0.001 0.01 0.0001)
# new weak_label_ratio refers to weak points/#raw_pc, has a 15 times relation
# TODO: normalize the ratios
# WEAK_LABEL_RATIOS=(0.015 0.15 0.0015) # corresponds to 0.001/0.01/0.0001

# TODO: ablation study
# num_k_query_pts
# how to concat features
# CONCAT_TYPES=('1234' '123' '234' '12' '1')
CONCAT_TYPES=('1234')


echo "training using WCE loss"
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

echo "training using WCE loss"
echo "finish training."