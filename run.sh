#!/bin/sh

#export CUDA_LAUNCH_BLOCKING=1 # for debugging

data_dir=./training_data/rea0
val_dir=./val_data
test_dir=./val_data

model=pix2pix

d_model=64

batch_size=4
n_epochs=8

#name=debug
name=${model}_${d_model}

output_dir=./output/${name}

mkdir -p $output_dir/checkpoints
mkdir -p $output_dir/test
mkdir -p ./tmp

norm=2.0e-7

#python main.py --name $name --isTrain --data_dir $data_dir --val_dir $val_dir --output_dir $output_dir --nrun 300 --nindex 100 --model $model --d_model $d_model --batch_size $batch_size --n_epochs $n_epochs --epoch_count 0 --print_freq 1 --save_latest_freq 10000 --save_image_freq 10000 --norm $norm #> ./tmp/out_${name}.log

python main.py --name $name --test_dir $test_dir --output_dir $output_dir --nrun 10 --model $model --d_model $d_model --load_iter -1 --norm $norm #> ./tmp/test_${name}.log



