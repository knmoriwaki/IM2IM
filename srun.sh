#!/bin/bash

#export CUDA_LAUNCH_BLOCKING=1 # for debugging

expected_hostname="cat4"
current_hostname=$(hostname)

if [ "$current_hostname" != "$expected_hostname" ]; then
    echo "Current host: $current_hostname"
    echo "Please login to $expected_hostname to execute this script."
    exit
else
    echo "Running IM2IM GAN pix2pix inference and XAI experiments on $current_hostname"
    test_python_env=infernce_venv
    xai_python_env=xai_venv

    val_dir=/mnt/data_cat4/moriwaki/IM2IM/val_data
    test_dir=/mnt/data_cat4/moriwaki/IM2IM/val_data
    results_dir=./output

    model=pix2pix_2
    name=added_forward__pix2pix_2_bs4_ep1_lambda1000_wgangp
    output_dir=./output/${name}

    norm=2.0e-7
    load_iter=-1
    igpu=1

    . ${test_python_env}/bin/activate
    which python
    # Test if all experiments still run
    # Normal testing
    python main.py --name $name --gpu_ids $igpu --test_dir $test_dir --output_dir $output_dir --results_dir $results_dir --nrun 100 --model $model --load_iter $load_iter --norm $norm 

    # XAI testing
    experiments=("ha" "oiii" "random" "random_ha" "random_oiii" "faint_ha" "occlusion")
    for xai_exp in "${experiments[@]}"
    do
        #python main.py --name $name --gpu_ids $igpu --test_dir $test_dir --output_dir $output_dir --results_dir $results_dir --nrun 100 --model $model --load_iter $load_iter --norm $norm --isXAI --xai_exp $xai_exp
        echo " Mocking the inference $xai_exp"
    done
    deactivate


    output_dir=./output
    results_dir=./output/xai_results
    . $xai_python_env/bin/activate
    which python
    # Evaluation of the inference on the test set
    #python eval.py --output_dir ${output_dir} --results_dir ${results_dir} --nrun 100 --isRef

    # XAI experiment evaluation
    for xai_exp in "${experiments[@]}"
    do
        #python eval.py --output_dir ${output_dir}  --results_dir ${results_dir} --nrun 100 --xai_exp xai_exp_${xai_exp}
        echo " Mocking the evaluation $xai_exp"
    done
    deactivate
fi




# Occlusion with sliding window
experiments=("occlusion")
stride=8
for xai_exp in "${experiments[@]}"
do
    #python main.py --name $name --gpu_ids $igpu --test_dir $test_dir --output_dir $output_dir --results_dir $results_dir --nrun 100 --model $model --load_iter $load_iter --norm $norm --isXAI --xai_exp $xai_exp --occlusion_stride $stride --occlusion_sample 71
    echo " Mocking the inference $xai_exp"
done


