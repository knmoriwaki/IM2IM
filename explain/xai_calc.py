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

def calc_importance(output_dir, ref_name, exp_name, total_n_occ, suffix, nbins=20, log_bins=True):
    """
    I want to assign an importance defied by the difference between 
    reference(r_(true-fake)) and experiment(r_(perturbed_true-fake)) to each patch in the occlusion
    """ 
    
    # Read reference data
    data = XAIDataLoader(output_dir, ref_name, suffix)
    ref_mix , _ = compute_r(data.real["obs"].values[0], data.fake["rec"].values[0], log_bins=log_bins)
    ref_ha , _ = compute_r(data.real["realA"].values[0], data.fake["fakeA"].values[0], log_bins=log_bins)
    ref_oiii , _ = compute_r(data.real["realB"].values[0], data.fake["fakeB"].values[0], log_bins=log_bins)

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
    occlusion_size = int(np.sqrt(im_size*im_size/total_n_occ))
    s = 0
    for i in range(0, rows, occlusion_size):
        for j in range(0, cols, occlusion_size):
            # Occluding the source with the masking values:
            im_mix[i:i + occlusion_size, j:j + occlusion_size] = l_mix[s]
            im_ha[i:i + occlusion_size, j:j + occlusion_size] = l_ha[s]
            im_oiii[i:i + occlusion_size, j:j + occlusion_size] = l_oiii[s]
            s += 1

    return im_mix, im_ha, im_oiii

## TO DO: Merge calc_eval_metrics and create_dataframe or write a wrapper around them
def calc_eval_metrics(data):
    """Calculate the evaluation metrics for the reconstructed maps of one data sample.
    1) Difference between the mean values (DBM). This resembles the L1 Norm, but we care about if the difference is positive or negative.
    2) Root mean square error (RMSE) between the true and reconstructed maps.
    3) Summation of the values in the true and reconstructed maps. Moreover, comparison (difference) between these two.
    4) The correlation coefficient after Fourier transformation.
    Input: data (dict) with following labels: ["obs", "trueHa", "trueOIII", "rec", "fakeHa", "fakeOIII"]
    Output: eval_dic (dict) with the evaluation metrics.
    """
    eval_dic = {}
    ## Error metrics
    eval_dic["l1_mix"] = np.mean(data["obs"]) - np.mean(data["rec"])
    eval_dic["l1_ha"] = np.mean(data["trueHa"]) - np.mean(data["fakeHa"])
    eval_dic["l1_oiii"] = np.mean(data["trueOIII"]) - np.mean(data["fakeOIII"])
    eval_dic["rmse_mix"] = np.sqrt(np.mean((data["obs"] - data["rec"])**2))
    eval_dic["rmse_ha"] = np.sqrt(np.mean((data["trueHa"] - data["fakeHa"])**2))
    eval_dic["rmse_oiii"] = np.sqrt(np.mean((data["trueOIII"] - data["fakeOIII"])**2))
    eval_dic["d_sum_mix"] = np.sum(data["obs"]) - np.sum(data["rec"])
    eval_dic["d_sum_ha"] = np.sum(data["trueHa"]) - np.sum(data["fakeHa"])
    eval_dic["d_sum_oiii"] = np.sum(data["trueOIII"]) - np.sum(data["fakeOIII"])
    ## Statistical metrics single field   
    eval_dic["mean_trueha"] = np.mean(data["trueHa"])
    eval_dic["mean_fakeha"] = np.mean(data["fakeHa"])
    eval_dic["mean_trueoiii"] = np.mean(data["trueOIII"])
    eval_dic["mean_fakeoiii"] = np.mean(data["fakeOIII"])
    eval_dic["std_trueha"] = np.std(data["trueHa"])
    eval_dic["std_fakeha"] = np.std(data["fakeHa"])
    eval_dic["std_trueoiii"] = np.std(data["trueOIII"])
    eval_dic["std_fakeoiii"] = np.std(data["fakeOIII"])
    eval_dic["sum_trueha"] = np.sum(data["trueHa"])
    eval_dic["sum_fakeha"] = np.sum(data["fakeHa"])
    eval_dic["sum_trueoiii"] = np.sum(data["trueOIII"])
    eval_dic["sum_fakeoiii"] = np.sum(data["fakeOIII"])
    eval_dic["max_trueha"] = np.max(data["trueHa"])
    eval_dic["max_fakeha"] = np.max(data["fakeHa"])
    eval_dic["max_trueoiii"] = np.max(data["trueOIII"])
    eval_dic["max_fakeoiii"] = np.max(data["fakeOIII"])


    # There are many correlation coefficients resulting from different k values. 
    # The k values for each map are the same, so we can just take the first one.
    r_mix , k_array = compute_r(data["obs"], data["rec"], log_bins=True)
    r_ha , _ = compute_r(data["trueHa"], data["fakeHa"], log_bins=True)
    r_oiii , _ = compute_r(data["trueOIII"], data["fakeOIII"], log_bins=True)
    i = 0
    for i in range(len(k_array)-1): 
        k = k_array[i]
        eval_dic["r_mix_"+str(int(k))]  = r_mix[i]
        eval_dic["r_ha_"+str(int(k))]   = r_ha[i]
        eval_dic["r_oiii_"+str(int(k))] = r_oiii[i]
    
    return eval_dic, k_array

def create_dataframe(output_dir, nrun=100, nindex=1):
    """
    Reads in several samples and creates a dataframe with the evaluation metrics.
    Input: suffix_list (list) with the ids of the samples.
    Output: df (pandas dataframe) with the evaluation metrics.
    """
    suffix_list = [ "run{:d}_index{:d}".format(i, j) for i in range(nrun) for j in range(nindex) ]
    data = read_data(output_dir, suffix=suffix_list[0], ldict=True)
    _, initial_k = calc_eval_metrics(data)
    temp = []
    for data_sample in suffix_list:
        data = read_data(output_dir, suffix=data_sample, ldict=True)
        # Perform analysis and calculate evaluation metrics
        eval_dic, k = calc_eval_metrics(data)
        assert k.all() == initial_k.all()
        # Add sample ID and timestamp to the evaluation results
        eval_dic['sample_id'] = data_sample
        temp.append(eval_dic)
    df = pd.DataFrame(temp)
    return df, initial_k

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