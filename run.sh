#!/bin/sh

#export CUDA_LAUNCH_BLOCKING=1 # for debugging

data_dir=./training_data
val_dir=./val_data
test_dir=./val_data

model=pix2pix

batch_size=8
n_epochs=2
n_epochs_decay=2
total_epochs=$(( n_epochs + n_epochs_decay ))
lambda=1000

#name=debug
name=${model}
name=test_${model}_${batch_size}_${total_epochs}

output_dir=./output/${name}

mkdir -p $output_dir/checkpoints
mkdir -p $output_dir/test
mkdir -p ./tmp

norm=2.0e-7

python main.py --name $name --isTrain --data_dir $data_dir --val_dir $val_dir --output_dir $output_dir --nrun 300 --nindex 100 --model $model --batch_size $batch_size --n_epochs $n_epochs --n_epochs_decay $n_epochs_decay --print_freq 1 --save_latest_freq 10000 --save_image_freq 1000 --norm $norm --lambda_L1 $lambda #> ./tmp/out_${name}.log

python main.py --name $name --test_dir $test_dir --output_dir $output_dir --nrun 100 --model $model --load_iter -1 --norm $norm #> ./tmp/test_${name}.log



