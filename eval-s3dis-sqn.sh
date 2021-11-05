#!/bin/bash

mode='test'
gpu=0
# 4 costs about 8GB GPU, others can be used on grp server GPUs
batch_size=(3) # 3 just 8GB,  4--> OOM on 1080 GPU
# 8 works on my nvidia 1080 gpu, others can be tried on grp server GPUs
val_batch_size=4 # 6-->system collapse
num_points=40960
max_epoch=100 # 400

LOG_NAMES=(
     'Log_weak_0.001_2021-11-01_16-25-48_WCE'
     'Log_weak_0.0001_2021-11-03_08-03-05_WCE'
     'Log_weak_2e-05_2021-11-04_15-31-17_WCE'
     'Log_weak_5e-05_2021-11-04_03-36-32_WCE'
     )

WEAK_LABEL_RATIOS=(
     0.001 
     0.0001 
     0.00005 
     0.00002
     ) 

# check above each item from LOG_NAMES's snapishots folder
EPOCHS=(
     36001
     35501
     27001
     23501
     )

for i in "${!LOG_NAMES[@]}"; do 

     echo "weak_label_ratio: ${WEAK_LABEL_RATIOS[$i]}"
     echo "model_path: results_Sqn/${LOG_NAMES[$i]}/snapshots/snap-${EPOCHS[$i]}"

     time python -B main_S3DIS_Sqn.py \
     --gpu ${gpu} \
     --mode ${mode} \
     --test_area 5 \
     --batch_size ${batch_size} \
     --val_batch_size ${val_batch_size} \
     --num_points ${num_points} \
     --max_epoch ${max_epoch} \
     --weak_label_ratio ${WEAK_LABEL_RATIOS[$i]} \
     --model_path results_Sqn/${LOG_NAMES[$i]}/snapshots/snap-${EPOCHS[$i]}

     echo "weak_label_ratio: ${WEAK_LABEL_RATIOS[$i]}"
     echo "model_path: results_Sqn/${LOG_NAMES[$i]}/snapshots/snap-${EPOCHS[$i]}"
done

echo "finish testing."