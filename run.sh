#!/bin/sh

#export CUDA_LAUNCH_BLOCKING=1 # for debugging

data_dir=./training_data
val_dir=./val_data
test_dir=./val_data

#model=pix2pix
model=pix2pix_2

batch_size=4
n_epochs=1
n_epochs_decay=0
total_epochs=$(( n_epochs + n_epochs_decay ))
lambda=100

#name=debug
name=${model}_bs${batch_size}_ep${total_epochs}_lambda${lambda}

output_dir=./output/${name}

mkdir -p $output_dir/checkpoints
mkdir -p $output_dir/test
mkdir -p ./tmp

norm=2.0e-7

igpu=0
python main.py --name $name --isTrain --gpu_ids $igpu --data_dir $data_dir --val_dir $val_dir --output_dir $output_dir --nrun 300 --nindex 100 --model $model --batch_size $batch_size --n_epochs $n_epochs --n_epochs_decay $n_epochs_decay --print_freq 1 --save_latest_freq 10000 --save_image_freq 1000 --norm $norm --lambda_L1 $lambda 

python main.py --name $name --gpu_ids $igpu --test_dir $test_dir --output_dir $output_dir --nrun 100 --model $model --load_iter -1 --norm $norm 



