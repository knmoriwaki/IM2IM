"""
Author: Scarlet Stadtler
Date: July 2023
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from explain.correlation_coefficient import compute_r
from explain.xai_dataloader import XAIDataLoader
import pdb

### Plotting routines ### 
def plot_true_fake_maps(data, results_dir, exp_name='test', suffix=f"run0_index0"):
    """ 
    Plots the true maps and the corresponding fake maps. Expects the data in form of the 
    DataFrame created by XAIDataLoader.
    
    """

    df_fake = data.fake
    df_real = data.real
    if data.pert is not None:
        # In case that the source data was pertubed we want to see the perturbed
        # data instead of the original (real) data
        df_real = data.pert 
    
    vmin = np.min(df_fake['fakeB'].values[0])
    #vmax = 9.0e-08
    vmax = np.max(df_fake['fakeA'].values[0])
    print("min", np.min(df_fake['fakeB'].values[0]))
    print(f"vmax: {vmax}")

    _, axs = plt.subplots(2,3, figsize=(10, 8))

    col = df_real.columns
    for i in range(len(col)):    
        ax = axs[0][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df_real[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)

    col = df_fake.columns
    for i in range(len(col)):    
        ax = axs[1][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df_fake[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)
        
    filename =f"true_fake_maps_{exp_name}_{suffix}_image.png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_diff_map(data, results_dir, exp_name, n_occ=None, suffix=f"run0_index0"):
    """
    Expects input from the XAIDataLoader
    """
    df_fake = data.fake
    df_real = data.real
    if data.pert is not None:
        # In case that the source data was pertubed we want to see the perturbed
        # data instead of the original (real) data
        df_real = data.pert 
    
    #vmin = -2.0e-07
    #vmax = 2.0e-07
    vmin = 0
    vmax = 9.0e-08  
    #vmax = np.max(df_real['obs'].values[0])
    
    _, axs = plt.subplots(3,3, figsize=(15, 8))
    
    col = df_real.columns
    for i in range(len(col)):    
        ax = axs[0][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df_real[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)

    col = df_fake.columns
    for i in range(len(col)):    
        ax = axs[1][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df_fake[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)

    # Create a diverging color map centered around zero
    cmap = plt.get_cmap('coolwarm')
    rcol = df_real.columns
    fcol = df_fake.columns
    for i in range(len(col)):    
        ax = axs[2][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        delta = df_real[rcol[i]].values[0] - df_fake[fcol[i]].values[0]
        im = ax.imshow(delta, interpolation="none", cmap=cmap)
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Difference')
    
    
    if n_occ is not None:
        filename =f"diff_map_{suffix}_{exp_name}_occluded{n_occ}_image.png"    
    else:
        filename =f"{exp_name}_{suffix}_image.png"
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path) 
    plt.show()
    plt.close()

def plot_perturbed_map(data, results_dir, exp_name, n_occ=None, suffix=f"run0_index0"):
    """
    Expects input from the XAIDataLoader
    """
    df_fake = data.fake
    df_real = data.real
    df_pert = data.pert
    
    #vmin = -2.0e-07
    #vmax = 2.0e-07
    vmin = 0
    vmax = 9.0e-08  
    #vmax = np.max(df_real['obs'].values[0])
    
    _, axs = plt.subplots(3,3, figsize=(10, 8))
    
    col = df_real.columns
    for i in range(len(col)):    
        ax = axs[0][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df_real[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)

    col = df_pert.columns
    for i in range(len(col)):    
        ax = axs[1][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df_pert[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)
    
    col = df_fake.columns
    for i in range(len(col)):    
        ax = axs[2][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df_fake[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)
    
    if n_occ is not None:
        filename =f"{exp_name}_{suffix}_occluded{n_occ}_image.png"    
    else:
        filename =f"{exp_name}_{suffix}_image.png"
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path) 
    plt.show()
    plt.close()

def plot_r_occ_sample(output_dir, ref_name, exp_name, results_dir, n_occ, suffix, nbins=20, log_bins=True):
    """
    Inputs: directories to read the data (output_dir, ref_name, exp_name) 
            and where to write the data (results_dir).
            n_occ: number of occluded images
    """
    plt.figure(figsize=(10, 6))
    
    # Plot occluded 
    l_mix = []
    l_ha = []
    l_oiii = []
    for i in range(int(n_occ)):
        data = XAIDataLoader(output_dir, exp_name, suffix, n_occ=i)
        
        r_mix , k_array = compute_r(data.pert["p_s"].values[0], data.fake["rec"].values[0], nbins=nbins, log_bins=log_bins)
        r_ha , _ = compute_r(data.pert["p_tA"].values[0], data.fake["fakeA"].values[0], nbins=nbins, log_bins=log_bins)
        r_oiii , _ = compute_r(data.pert["p_tB"].values[0], data.fake["fakeB"].values[0], nbins=nbins, log_bins=log_bins)
        k = k_array[0:-1]
        plt.plot(k, r_mix, 'k', alpha=0.01)
        plt.plot(k, r_ha, 'b', alpha=0.05)
        plt.plot(k, r_oiii, 'r', alpha=0.05)
        l_mix.append(r_mix)
        l_ha.append(r_ha)
        l_oiii.append(r_oiii)
        
    mean_mix = np.mean(l_mix, axis=0)
    mean_ha = np.mean(l_ha, axis=0)
    mean_oiii = np.mean(l_oiii, axis=0)
    plt.plot(k, mean_mix, '--', color='k', label="mixed signal")
    plt.plot(k, mean_ha, '--', color='b', label="Ha signal")
    plt.plot(k, mean_oiii, '--', color='r', label="OIII signal")
    
    # Plot reference
    data = XAIDataLoader(output_dir, ref_name, suffix)
    r_mix , _ = compute_r(data.real["obs"].values[0], data.fake["rec"].values[0], nbins=nbins, log_bins=log_bins)
    r_ha , _ = compute_r(data.real["realA"].values[0], data.fake["fakeA"].values[0], nbins=nbins, log_bins=log_bins)
    r_oiii , _ = compute_r(data.real["realB"].values[0], data.fake["fakeB"].values[0], nbins=nbins, log_bins=log_bins)
    
    plt.plot(k, r_mix, color='k', label="ref mix")
    plt.plot(k, r_ha, color='b', label="ref Ha")
    plt.plot(k, r_oiii, color='r', label="ref OIII")
    
    plt.legend()
    plt.xlabel("k in log bins")
    plt.xscale('log')
    plt.ylabel("r between true and fake "+suffix)
    plt.title(suffix)
    plt.savefig(f"{results_dir}/compare_occlusion_exp_{suffix}.png")
    print(f"Saved plot {results_dir}/compare_occlusion_exp_{suffix}.png")
    plt.show()
    plt.close()

def plot_occlusion_sensitivity(im_mix, im_ha, im_oiii, results_dir, sscale):
    # reproduced map
    label_list = ["mix", "Ha", "OIII"]
    vmin = 0.0
    vmax = 3.0

    fig, axs = plt.subplots(1,3, figsize=(10, 6))

    ax = axs[0]
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax.set_title(label_list[0])
    im = ax.imshow(im_mix, interpolation="none")#, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    
    ax = axs[1]
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax.set_title(label_list[1])
    im = ax.imshow(im_ha, interpolation="none")#, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    
    ax = axs[2]
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax.set_title(label_list[2])
    im = ax.imshow(im_oiii, interpolation="none")#, vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    #cbar.set_label('Sum difference r')
    # Adjust spacing between subplots
    plt.tight_layout()
    
    filename ="occlusion_sensitivity_image"+sscale+".png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path)
    print(f"Saved plot {save_path}")
    plt.show()
    plt.close()

def plot_all_r_vs_k(r_list, results_dir, title="Insert Title"):
    """ Plots r vs k for all test samples in one plot.
    """
    mean_r = np.mean(np.array((r_list[1:])), axis=0)
    k = r_list[0]
    r = r_list[1:]
    
    plt.figure(figsize=(10, 6))
    for i in range(len(r)):
        plt.plot(k, r[i], 'c', alpha=0.1)
    plt.plot(k, mean_r, 'r', label="mean r")
    plt.legend()
    plt.xlabel("k in log bins")
    plt.xscale('log')
    plt.ylabel("r between reference and experiment")
    plt.title(title)
    name = '_'.join(title.lower().split()).replace(' ', '_')
    plt.savefig(f"{results_dir}/compare_exp{name}.png")
    print(f"Saved plot {results_dir}/{name}.png")
    plt.show()
    plt.close()

def plot_r_single_sample(data, results_dir, suffix, nbins=20, log_bins=True):
    """
    Plot the r values for a single sample.
    Input: data (pandas dataframe) with the real and fake values.
    """

    df_fake = data.fake
    df_real = data.real
    m = df_real["obs"].values[0]
    a = df_real["realA"].values[0]
    b = df_real["realB"].values[0]
    if data.pert is not None:
        # In case that the source data was pertubed we want to see the perturbed
        # data instead of the original (real) data
        df_pert = data.pert 
        m = df_pert["p_s"].values[0]
        a = df_pert["p_tA"].values[0]
        b = df_pert["p_tB"].values[0]

    r_mix , k_array = compute_r(m, df_fake["rec"].values[0], nbins=nbins, log_bins=log_bins)
    r_ha , _ = compute_r(a, df_fake["fakeA"].values[0], nbins=nbins, log_bins=log_bins)
    r_oiii , _ = compute_r(b, df_fake["fakeB"].values[0], nbins=nbins, log_bins=log_bins)
    
    k = k_array[0:-1]
        
    plt.figure(figsize=(10, 6))
    plt.plot(k, r_mix, 'k', label="r mixed signal")
    plt.plot(k, r_ha, 'b', label="r Halpha signal")
    plt.plot(k, r_oiii, 'r', label="r OIII signal")
    plt.legend()
    plt.xlabel("k in log bins")
    plt.xscale('log')
    plt.ylabel("r between true and fake "+suffix)
    plt.title(suffix)
    plt.savefig(f"{results_dir}/r_single_sample_{suffix}.png")
    print(f"Saved plot {results_dir}/{suffix}.png")
    plt.show()
    plt.close()


def plot_two_models_r_vs_k(r_mix_list, r_ha_list, r_oiii_list, r_mix_list2, r_ha_list2, r_oiii_list2, results_dir, mod="None", title="Insert Title"):
    """ 
    Plots r vs k for all test samples in one plot for two GAN models (or two experiments).
    """
    def split_mean(r_list):
        mean_r = np.mean(np.array((r_list[1:])), axis=0)
        k = r_list[0]
        r = r_list[1:]
        return mean_r, r, k
    
    m_mix, r_mix, k = split_mean(r_mix_list)
    m_ha, r_ha, _ = split_mean(r_ha_list)
    m_oiii, r_oiii, _ = split_mean(r_oiii_list)
    
    m_mix2, r_mix2, _ = split_mean(r_mix_list2)
    m_ha2, r_ha2, _ = split_mean(r_ha_list2)
    m_oiii2, r_oiii2, _ = split_mean(r_oiii_list2)
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
    
    ax = axs[0]
    for i in range(len(r_mix)):
        ax.plot(k, r_mix[i], 'forestgreen', alpha=0.01)
    ax.plot(k, m_mix, 'forestgreen', label="original")
    for i in range(len(r_mix2)):
        ax.plot(k, r_mix2[i],  color='lime', alpha=0.01)
    ax.plot(k, m_mix2, '--', color='lime', label=mod)
    ax.legend()
    ax.set_ylabel('r mix')
    ax.set_xscale('log')
    ax.set_title('True vs. Fake for two different models')
    
    ax = axs[1]
    for i in range(len(r_ha)):
        ax.plot(k, r_ha[i], 'b', alpha=0.01)
    ax.plot(k, m_ha, 'b', label="original")
    for i in range(len(r_mix2)):
        ax.plot(k, r_ha2[i],  color='c', alpha=0.01)
    ax.plot(k, m_ha2, '--', color='c', label=mod)
    ax.legend()
    ax.set_ylabel('r Ha')
    ax.set_xscale('log')
    
    ax = axs[2]
    for i in range(len(r_oiii)):
        ax.plot(k, r_oiii[i], 'r', alpha=0.01)
    ax.plot(k, m_oiii, 'r', label="original")
    for i in range(len(r_mix2)):
        ax.plot(k, r_oiii2[i],  color='orange', alpha=0.01)
    ax.plot(k, m_oiii2, '--', color='orange', label=mod)
    ax.legend()
    ax.set_ylabel('r OIII')
    ax.set_xscale('log')
    ax.set_xlabel('k in log bins')
    
    name = '_'.join(title.lower().split()).replace(' ', '_')
    plt.savefig(f"{results_dir}/compare_exp{name}.png")
    print(f"Saved plot {results_dir}/{name}.png")
    plt.show()
    plt.close()




if __name__ == "__main__":
    output_dir = "../output"
    names = ['test', 'xai_exp_occlusion']
    results_dir = "../output/xai_occlusion_results/"
    nrun = 0
    nindex = 1
    total_n_occ = 16
    suffix=f"run71_index0"

    df = XAIDataLoader(output_dir, names[1], suffix, n_occ=5)
    plot_diff_map(df, results_dir, names[1], n_occ=5, suffix=suffix)