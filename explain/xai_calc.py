"""
Author: Scarlet Stadtler
Date: July 2023
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy.io import fits
from correlation_coefficient import compute_r
from xai_dataloader import XAIDataLoader
from xai_plot import plot_occlusion_sensitivity
import pdb

def calc_importance(output_dir, ref_name, exp_name, total_n_occ, suffix, sscale=None, 
                    occlusion_size=64, stride=32, nbins=20, log_bins=True):
    """
    I want to assign an importance defied by the difference between 
    reference(r_(true-fake)) and experiment(r_(perturbed_true-fake)) to each patch in the occlusion
    """ 
    
    # Read reference data
    data = XAIDataLoader(output_dir, ref_name, suffix)
    ref_mix , k = compute_r(data.real["obs"].values[0], data.fake["rec"].values[0], log_bins=log_bins)
    ref_ha , _ = compute_r(data.real["realA"].values[0], data.fake["fakeA"].values[0], log_bins=log_bins)
    ref_oiii , _ = compute_r(data.real["realB"].values[0], data.fake["fakeB"].values[0], log_bins=log_bins)
    # Choose spatial scale
    # Current k vector is expected to look like this (20 bins, log-spaced):
    # [ 6.28318531    8.14828324   10.567016     13.70372433   17.77153175
    #  23.04682532   29.88803469   38.7599856    50.26548246   65.18626588
    #  84.53612802  109.62979461  142.17225402  184.37460259  239.10427751
    #  310.07988476  402.12385966  521.49012708  676.28902415  877.0383569
    #  1137.37803217 ]
    if sscale=="small":
        k_start = 11 # k[11] = 109.62979461196917
        k_stop  = 20 # k[20] = 877.0383569, not k[21], because we usually drop k[21]
    elif sscale=="large":
        k_start = 0 # k[0] = 6.28318531
        k_stop  = 5 # k[3] = 13.70372433
    else:
        k_start = 0
        k_stop  = 20 # not k[21], because we usually drop k[21]
    
    # Restrict to only the selected bins for large/small/all scale structures
    ref_mix = ref_mix[k_start:k_stop]
    ref_ha = ref_ha[k_start:k_stop]
    ref_oiii = ref_oiii[k_start:k_stop]

    im_size = 256

    l_mix = []
    l_ha = []
    l_oiii = []
    for i in range(total_n_occ):
        # Read occluded data
        data = XAIDataLoader(output_dir, exp_name, suffix, n_occ=i)
        df_p = data.pert
        df_f = data.fake
        # Calculate correlation coefficients
        r_mix , _ = compute_r(df_p["p_s"].values[0], df_f["rec"].values[0], nbins=nbins, log_bins=log_bins)
        r_ha , _ = compute_r(df_p["p_tA"].values[0], df_f["fakeA"].values[0], nbins=nbins, log_bins=log_bins)
        r_oiii , _ = compute_r(df_p["p_tB"].values[0], df_f["fakeB"].values[0], nbins=nbins, log_bins=log_bins)
        # Restrict to only the first 10 bins for large scale structures
        r_mix = r_mix[k_start:k_stop]
        r_ha = r_ha[k_start:k_stop]
        r_oiii = r_oiii[k_start:k_stop]
        # Calculate "Score"
        ds_mix = np.sum(abs(ref_mix - r_mix))
        ds_ha = np.sum(abs(ref_ha - r_ha))
        ds_oiii = np.sum(abs(ref_oiii - r_oiii))
        # Store
        l_mix.append(ds_mix)
        l_ha.append(ds_ha)
        l_oiii.append(ds_oiii)
       
        
    im_mix  = np.zeros((im_size,im_size))
    im_ha   = np.zeros((im_size,im_size))
    im_oiii = np.zeros((im_size,im_size))
    counter = np.zeros((im_size,im_size))
    
    rows, cols = np.shape(im_mix)

    padding = occlusion_size - stride  # Calculate the padding size
    pad_left = padding
    pad_right = padding
    pad_top = padding
    pad_bottom = padding
    counter = np.pad(counter, ( (pad_top, pad_bottom), (pad_left, pad_right) ), constant_values=(100, 100))
    im_mix = np.pad(im_mix, ( (pad_top, pad_bottom), (pad_left, pad_right) ), constant_values=(100, 100))
    im_ha = np.pad(im_ha, ( (pad_top, pad_bottom), (pad_left, pad_right) ), constant_values=(100, 100))
    im_oiii = np.pad(im_oiii, ( (pad_top, pad_bottom), (pad_left, pad_right) ), constant_values=(100, 100))

    rows, cols = np.shape(counter)
    
    s = 0
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            # Occluding the source with the masking values:
            im_mix[i:i + occlusion_size, j:j + occlusion_size] += l_mix[s]
            im_ha[i:i + occlusion_size, j:j + occlusion_size] += l_ha[s]
            im_oiii[i:i + occlusion_size, j:j + occlusion_size] += l_oiii[s]
            s += 1
            counter[i:i+occlusion_size, j:j+occlusion_size] += 1.

    counter = counter[pad_top:pad_top+im_size, pad_left:pad_left+im_size]
    im_mix = im_mix[pad_top:pad_top+im_size, pad_left:pad_left+im_size]
    im_ha = im_ha[pad_top:pad_top+im_size, pad_left:pad_left+im_size]
    im_oiii = im_oiii[pad_top:pad_top+im_size, pad_left:pad_left+im_size]

    im_mix = im_mix / counter
    im_ha = im_ha / counter
    im_oiii = im_oiii / counter

    return im_mix, im_ha, im_oiii

def calc_eval_metrics(data, compare_to="real", nbins=20, log_bins=True):
    """Calculate the evaluation metrics for the reconstructed maps of one data sample.
    1) Difference between the mean values (DBM). This resembles the L1 Norm, but we care about if the difference is positive or negative.
    2) Root mean square error (RMSE) between the true and reconstructed maps.
    3) Summation of the values in the true and reconstructed maps. Moreover, comparison (difference) between these two.
    4) The correlation coefficient after Fourier transformation.
    Input: XAIDataLoader class 
    Output: eval_dic (dict) with the evaluation metrics as input to create_eval_dataframe
    """
    
    df_fake = data.fake
    f_mix = df_fake["rec"].values[0]
    f_A = df_fake["fakeA"].values[0]
    f_B = df_fake["fakeB"].values[0]
    
    if compare_to == "real":
        df_ref = data.real
        r_mix = df_ref["obs"].values[0]
        r_A = df_ref["realA"].values[0]
        r_B = df_ref["realB"].values[0]
    elif compare_to == "pert":
        df_ref = data.pert
        r_mix = df_ref["p_s"].values[0]
        r_A = df_ref["p_tA"].values[0]
        r_B = df_ref["p_tB"].values[0]
    else:
        print("Currently it is default to compare to the real data.")
        print("If you do not want to compare to real or perturbed (pert) data extend the function.")
    
    eval_dic = {}
    ## Error metrics
    eval_dic["l1_mix"] = np.mean(r_mix) - np.mean(f_mix)
    eval_dic["l1_ha"] = np.mean(r_A) - np.mean(f_A)
    eval_dic["l1_oiii"] = np.mean(r_B) - np.mean(f_B)
    eval_dic["rmse_mix"] = np.sqrt(np.mean((r_mix - f_mix)**2))
    eval_dic["rmse_ha"] = np.sqrt(np.mean((r_A - f_A)**2))
    eval_dic["rmse_oiii"] = np.sqrt(np.mean((r_B - f_B)**2))
    eval_dic["d_sum_mix"] = np.sum(r_mix) - np.sum(f_mix)
    eval_dic["d_sum_ha"] = np.sum(r_A) - np.sum(f_A)
    eval_dic["d_sum_oiii"] = np.sum(r_B) - np.sum(f_B)
    ## Statistical metrics single field   
    eval_dic["mean_trueha"] = np.mean(r_A)
    eval_dic["mean_fakeha"] = np.mean(f_A)
    eval_dic["mean_trueoiii"] = np.mean(r_B)
    eval_dic["mean_fakeoiii"] = np.mean(f_B)
    eval_dic["std_trueha"] = np.std(r_A)
    eval_dic["std_fakeha"] = np.std(f_A)
    eval_dic["std_trueoiii"] = np.std(r_B)
    eval_dic["std_fakeoiii"] = np.std(f_B)
    eval_dic["sum_trueha"] = np.sum(r_A)
    eval_dic["sum_fakeha"] = np.sum(f_A)
    eval_dic["sum_trueoiii"] = np.sum(r_B)
    eval_dic["sum_fakeoiii"] = np.sum(f_B)
    eval_dic["max_trueha"] = np.max(r_A)
    eval_dic["max_fakeha"] = np.max(f_A)
    eval_dic["max_trueoiii"] = np.max(r_B)
    eval_dic["max_fakeoiii"] = np.max(f_B)


    # There are many correlation coefficients resulting from different k values. 
    # The k values for each map are the same, so we can just take the first one.
    r_mix , k_array = compute_r(r_mix, f_mix, nbins=nbins, log_bins=log_bins)
    r_ha , _ = compute_r(r_A, f_A, nbins=nbins, log_bins=log_bins)
    r_oiii , _ = compute_r(r_B, f_B, nbins=nbins, log_bins=log_bins)
    i = 0
    for i in range(len(k_array)-1): 
        k = k_array[i]
        eval_dic["r_mix_"+str(int(k))]  = r_mix[i]
        eval_dic["r_A_"+str(int(k))]   = r_ha[i]
        eval_dic["r_B_"+str(int(k))] = r_oiii[i]
    
    return eval_dic, k_array

def create_eval_dataframe(output_dir, exp_name, suffix_list, nbins=20, log_bins=True):
    """
    Reads in several samples and creates a dataframe with the evaluation metrics.
    Input: suffix_list (list) with the ids of the samples.
    Output: df (pandas dataframe) with the evaluation metrics.
    """
    temp = []
    for data_sample in suffix_list:
        data = XAIDataLoader(output_dir, exp_name, suffix=data_sample)
        # Perform analysis and calculate evaluation metrics
        eval_dic, k = calc_eval_metrics(data, compare_to='real', nbins=nbins, log_bins=log_bins)
        # Add sample ID and timestamp to the evaluation results
        eval_dic['sample_id'] = data_sample
        temp.append(eval_dic)
    df = pd.DataFrame(temp)
    return df, k

def compare_experiments(data_ref, data_exp, nbins=20, log_bins=True, ldict=False):
    """
    Compare the correlation coefficients of two experiments.
    """
    r_mix , k_array = compute_r(data_ref["rec"].values[0], data_exp["rec"].values[0], nbins=nbins, log_bins=log_bins)
    r_ha , _ = compute_r(data_ref["fakeA"].values[0], data_exp["fakeA"].values[0], nbins=nbins, log_bins=log_bins)
    r_oiii , _ = compute_r(data_ref["fakeB"].values[0], data_exp["fakeB"].values[0], nbins=nbins, log_bins=log_bins)

    if ldict: 
        r = {}
        for i in range(len(k_array)-1): 
            k = k_array[i]
            r["r_mix_"+str(int(k))]  = r_mix[i]
            r["r_ha_"+str(int(k))]   = r_ha[i]
            r["r_oiii_"+str(int(k))] = r_oiii[i]
        
        return r_ha, k_array[0:-1]
    else:
        return r_mix, r_ha, r_oiii, k_array[0:-1]

def compare_exp_testset(output_dir, ref_name, exp_name, nrun=100, nindex=1, nbins=20, log_bins=True):
    """ Construct a list for plotting the correlation r between two experiments against k for the whole test dataset.
    Inputs: data directories for both experiments (ref_dir and exp_dir)
            log_bins switch in case the computation of r should run on log(k)
            nrun ask Kana
            nindex ask Kana
    Outputs: three lists containing the k as first entry and then correlation coefficients correspoding to k 
    for each sample in the test set
    """
    
    suffix_list = [ "run{:d}_index{:d}".format(i, j) for i in range(nrun) for j in range(nindex) ]
    r_mix_list  = []
    r_ha_list   = []
    r_oiii_list = []
    
    for data_sample in suffix_list:
        data_ref = XAIDataLoader(output_dir, ref_name, data_sample).fake
        data_exp = XAIDataLoader(output_dir, exp_name, data_sample).fake
        r_mix, r_ha, r_oiii, k = compare_experiments(data_ref, data_exp, nbins=nbins, log_bins=log_bins, ldict=False)
        if data_sample == suffix_list[0]:
            r_mix_list.append(k)
            r_ha_list.append(k)
            r_oiii_list.append(k)
        r_mix_list.append(r_mix)
        r_ha_list.append(r_ha)
        r_oiii_list.append(r_oiii)
        
    return r_mix_list, r_ha_list, r_oiii_list

def occlusion(occlusion_size=64, stride=32):

    im_size = 256
    counter = np.zeros((im_size,im_size))

    padding = occlusion_size - stride  # Calculate the padding size
    pad_left = padding
    pad_right = padding
    pad_top = padding
    pad_bottom = padding
    counter = np.pad(counter, ( (pad_top, pad_bottom), (pad_left, pad_right) ), constant_values=(100, 100))
    rows, cols = np.shape(counter)
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            counter[i:i+occlusion_size, j:j+occlusion_size] += 1. #np.ones((occlusion_size, occlusion_size))

    counter = counter[pad_top:im_size, pad_left:im_size] # Why does this work?
    return counter

def plot_counter(counter, results_dir, name):
    print(np.max(counter))
    im = plt.imshow(counter, interpolation="none")
    plt.colorbar(im, orientation='horizontal', pad=0.2)
    filename =str(name)+"_counter.png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path)
    print(f"Saved plot {save_path}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    output_dir = "../output/original_GAN_xai_experiments"
    names = ['test', 'xai_exp_occlusion']
    results_dir = "../output/original_GAN_xai_experiments/xai_occlusion_results/"
    nrun = 0
    nindex = 1
    total_n_occ = 16
    suffix=f"run71_index0"

    ## start timing 
    start = time.time()
    counter = occlusion(occlusion_size=64, stride=32)
    #for i in range(60):
    #    plot_counter(counter[i], results_dir, i)
    plot_counter(counter, results_dir, 'outside-loop-padding')

    #im_mix, im_ha, im_oiii = calc_importance(output_dir, names[0], names[1], 
    #                                         1024, suffix, sscale="large", stride=8, nbins=20, log_bins=True)
    
    #plot_occlusion_sensitivity(im_mix, im_ha, im_oiii, results_dir, "large") 

    #im_mix, im_ha, im_oiii = calc_importance(output_dir, names[0], names[1], 
    #                                         1024, suffix, sscale="small", stride=8, nbins=20, log_bins=True)
    
    #plot_occlusion_sensitivity(im_mix, im_ha, im_oiii, results_dir, "small") 

    #im_mix, im_ha, im_oiii = calc_importance(output_dir, names[0], names[1], 
    #                                         1024, suffix, sscale=None, stride=8, nbins=20, log_bins=True)
    
    #plot_occlusion_sensitivity(im_mix, im_ha, im_oiii, results_dir, "all_scales") 
    print(time.time() - start, "seconds elapsed")