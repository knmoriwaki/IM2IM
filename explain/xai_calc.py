"""
Author: Scarlet Stadtler
Date: July 2023
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from correlation_coefficient import compute_r
from xai_dataloader import XAIDataLoader
import pdb

def calc_importance(output_dir, ref_name, exp_name, total_n_occ, suffix, occlusion_size=64, stride=32, nbins=20, log_bins=True):
    """
    I want to assign an importance defied by the difference between 
    reference(r_(true-fake)) and experiment(r_(perturbed_true-fake)) to each patch in the occlusion
    """ 
    
    # Read reference data
    data = XAIDataLoader(output_dir, ref_name, suffix)
    ref_mix , _ = compute_r(data.real["obs"].values[0], data.fake["rec"].values[0], log_bins=log_bins)
    ref_ha , _ = compute_r(data.real["realA"].values[0], data.fake["fakeA"].values[0], log_bins=log_bins)
    ref_oiii , _ = compute_r(data.real["realB"].values[0], data.fake["fakeB"].values[0], log_bins=log_bins)
    # Restrict to only the first 10 bins for large scale structures
    ref_mix = ref_mix[:10]
    ref_ha = ref_ha[:10]
    ref_oiii = ref_oiii[:10]
    

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
        r_mix = r_mix[:10]
        r_ha = r_ha[:10]
        r_oiii = r_oiii[:10]
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
    
    rows, cols = np.shape(im_mix)

    s = 0
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            # Occluding the source with the masking values:
            im_mix[i:i + occlusion_size, j:j + occlusion_size] += l_mix[s]
            im_ha[i:i + occlusion_size, j:j + occlusion_size] += l_ha[s]
            im_oiii[i:i + occlusion_size, j:j + occlusion_size] += l_oiii[s]
            s += 1
    im_mix = im_mix / np.max(im_mix)
    im_ha = im_ha / np.max(im_ha)
    im_oiii = im_oiii / np.max(im_oiii)

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