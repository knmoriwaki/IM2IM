#!/bin/bash

#export CUDA_LAUNCH_BLOCKING=1 # for debugging

val_dir=/mnt/data_cat4/moriwaki/IM2IM/val_data
test_dir=/mnt/data_cat4/moriwaki/IM2IM/val_data
results_dir=./output

model=pix2pix_2
name=pix2pix_2_bs4_ep1_lambda1000_vanilla
output_dir=/mnt/data_cat4/moriwaki/IM2IM/output/${name}

norm=2.0e-7
load_iter=-1
igpu=1

# Test if all experiments still run
# Normal testing
#python main.py --name $name --gpu_ids $igpu --test_dir $test_dir --output_dir $output_dir --results_dir $results_dir --nrun 100 --model $model --load_iter $load_iter --norm $norm 

# XAI testing
experiments=("ha" "oiii" "random" "random_ha" "random_oiii" "faint_ha" "occlusion")
for xai_exp in "${experiments[@]}"
do
    python main.py --name $name --gpu_ids $igpu --test_dir $test_dir --output_dir $output_dir --results_dir $results_dir --nrun 100 --model $model --load_iter $load_iter --norm $norm --isXAI --xai_exp $xai_exp
done

# Occlusion with sliding window
experiments=("occlusion")
stride=8
for xai_exp in "${experiments[@]}"
do
    python main.py --name $name --gpu_ids $igpu --test_dir $test_dir --output_dir $output_dir --results_dir $results_dir --nrun 100 --model $model --load_iter $load_iter --norm $norm --isXAI --xai_exp $xai_exp --occlusion_stride $stride --occlusion_sample 71
done


