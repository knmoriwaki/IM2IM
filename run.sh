#!/bin/bash

#export CUDA_LAUNCH_BLOCKING=1 # for debugging

data_dir=./training_data
test_dir=./val_data

batch_size=64
n_epochs=10
lambda=10

gan_mode=vanilla
gan_mode=wgan

name=_bs${batch_size}_ep${n_epochs}_lambda${lambda}_${gan_mode}

output_dir=./output/${name}

mkdir -p $output_dir/checkpoints
mkdir -p $output_dir/test
mkdir -p ./tmp

igpu=1

python3 main.py --name $name --gpu_ids $igpu --data_dir $data_dir --output_dir $output_dir --batch_size $batch_size --n_epochs $n_epochs --print_freq 1 --save_latest_freq 10000 --save_image_freq 1000 --lambda_L1 $lambda --gan_mode $gan_mode --dropout 0.8 --lambda_z 2 #--hidden_dim_G 16 --hidden_dim_D 16




