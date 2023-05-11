#!/bin/sh

#export CUDA_LAUNCH_BLOCKING=1 # for debugging

data_dir=./training_data/rea0
val_dir=./val_data
test_dir=./test_data

model=pix2pix

d_model=32

batch_size=4
n_epochs=8

#name=debug
name=${model}_${d_model}

output_dir=./output/${name}

mkdir -p $output_dir/checkpoints
mkdir -p $output_dir/test
mkdir -p ./tmp


python main.py --name $name --isTrain --data_dir $data_dir --val_dir $val_dir --output_dir $output_dir --nrun 300 --nindex 1 --model $model --d_model $d_model --batch_size $batch_size --n_epochs $n_epochs --epoch_count 0 --print_freq 1 --save_latest_freq 100 #> ./tmp/out_${name}.log

python main.py --name $name --test_dir $data_dir --output_dir $output_dir --nrun 10 --model $model --d_model $d_model --load_iter -1 #> ./tmp/test_${name}.log



