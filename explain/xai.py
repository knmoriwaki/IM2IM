import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from correlation_coefficient import compute_r
from xai_dataloader import XAIDataLoader
import pdb

def save_fits_data(image, path, norm=2.0e-7, overwrite=False):
    ## save astropy.io.fits 2d image data
    # image: np.array((1,1,N,N)) or torch.tensor((1,1,N,N)). The first (batch) and second (feature) dimensions will be squeezed.
    # path: output file name 
    # norm: set the same normalization factor used in load_data
    # overwrite: False in default
    img = image.squeeze()
    img = norm * img #Scarlet is this correct? There was no variable called "fac"
    
    hdu = fits.PrimaryHDU(img)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=overwrite)

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
    
    vmin = 0
    vmax = 9.0e-08
    #vmax = np.max(df_real['obs'].values[0])

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

def plot_shuffled_map(df, results_dir, exp_name='test', suffix=f"run0_index0"):
    #vmin = -2.0e-07
    #vmax = 2.0e-07
    vmin = 0
    vmax = 9.0e-08  
    
    _, axs = plt.subplots(3,3, figsize=(10, 8))
    
    col = df.columns
    for i in range(len(col)):
        ax = axs[int(i/3)][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(col[i])
        ax.imshow(df[col[i]].values[0], interpolation="none", vmin=vmin, vmax=vmax)
        
    filename =f"{exp_name}_{suffix}_image.png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path) 
    plt.show()
    plt.close()


def plot_occluded_map(data, results_dir, n_occ, exp_name, suffix=f"run0_index0"):
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
    
    filename =f"{exp_name}_{suffix}_occluded{n_occ}_image.png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path) 
    plt.show()
    plt.close()

def plot_r_occ_sample(output_dir, ref_name, exp_name, results_dir, n_occ, suffix, nbins=20, log_bins=True):
    """
    Inputs: df (dataframe holding the real, perturbed (occluded) and generated images)
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
        plt.plot(k, r_mix, 'k', alpha=0.05)
        plt.plot(k, r_ha, 'b', alpha=0.1)
        plt.plot(k, r_oiii, 'r', alpha=0.1)
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
    r_mix , _ = compute_r(data.real["obs"].values[0], data.fake["rec"].values[0], log_bins=log_bins)
    r_ha , _ = compute_r(data.real["realA"].values[0], data.fake["fakeA"].values[0], log_bins=log_bins)
    r_oiii , _ = compute_r(data.real["realB"].values[0], data.fake["fakeB"].values[0], log_bins=log_bins)
    
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

def plot_occlusion_sensitivity(im_mix, im_ha, im_oiii, results_dir):
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
    
    filename ="occlusion_sensitivity_image.png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path)
    plt.show()

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


def write_zero_fits(output_dir, data):
    #### Actually this function is not needed. For the experiment I just multiplied the tensors directly with zero!
    """Write fits files with zero values for the reconstructed maps.
    Input: output_dir (str) with the path to the output directory.
           data (dict) with following labels: ["obs", "trueHa", "trueOIII", "rec", "fakeHa", "fakeOIII"]
    Output: fits file
    """
    zeros = np.zeros(data['fakeOIII'].shape)
    size = zeros.shape
    img = zeros.reshape(1, 1, size[0], size[1])
    suffix = "no_signal"
    file_name = f"{output_dir}/{suffix}.fits"
    save_fits_data(img, file_name, norm=2.0e-7, overwrite=False)

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

def plot_true_vs_k(df, k_array, results_dir, moment="mean", exp_name="yolo"):
    """
    Plot the mean of the maps vs k.
    Input: df (pandas dataframe) with the evaluation metrics.
           k_array (numpy array) with the k values.
           output_dir (str) with the path to the output directory.
           name (str) with the name of the plot.
    Output: plot
    """
    for i in range(len(k_array)-1): 
        k = int(k_array[i])
        plt.figure(figsize=(10, 6))
        plt.plot(df[moment+"_trueha"], df["r_ha_"+str(k)], label="Halpha", marker='o', linestyle='None')
        plt.plot(df[moment+"_trueoiii"], df["r_oiii_"+str(k)], label="OIII", marker='o', linestyle='None')
        plt.legend()
        plt.xlabel(moment+" of true signal")
        plt.ylabel("r between true and fake at k= "+str(k))
        name = exp_name+"_"+moment+"_vs_k_"+str(k)
        plt.title(name)
        plt.savefig(f"{results_dir}/{name}.png")
        print(f"Saved plot {results_dir}/{name}.png")
        plt.show()
        plt.close()

def plot_k_vs_error(df, k_array, results_dir, error="l1", exp_name="yolo"):
    """
    Plot the mean of the maps vs k.
    Input: df (pandas dataframe) with the evaluation metrics.
           k_array (numpy array) with the k values.
           output_dir (str) with the path to the output directory.
           name (str) with the name of the plot.
    Output: plot
    """
    for i in range(len(k_array)-1): 
        k = int(k_array[i])
        plt.figure(figsize=(10, 6))
        plt.plot(df["r_ha_"+str(k)], df[error+"_ha"], label="Halpha", marker='o', linestyle='None')
        plt.plot(df["r_oiii_"+str(k)], df[error+"_oiii"], label="OIII", marker='o', linestyle='None')
        plt.legend()
        plt.ylabel(error+" of true and fake signal")
        plt.xlabel("r between true and fake at k= "+str(k))
        name = exp_name+"_"+error+"_vs_k_"+str(k)
        plt.title(name)
        plt.savefig(f"{results_dir}/{name}.png")
        print(f"Saved plot {results_dir}/{name}.png")
        plt.show()
        plt.close()

def plot_sample_r_vs_k(data_ref, data_exp, results_dir, title="Insert Title"):
    """ This plot needs improvement to plot all data not only one data point.
    """
    r_mix, r_ha, r_oiii, k = compare_experiments(data_ref, data_exp, log_bins=True, ldict=False)
    plt.figure(figsize=(10, 6))
    plt.plot(k, r_mix, 'r', label="reconstructed mixed signal")
    plt.plot(k, r_ha, 'b', label="reconstructed Halpha signal")
    plt.plot(k, r_oiii, 'g', label="reconstructed OIII signal")
    plt.xscale('log')
    plt.legend()
    plt.xlabel("k in log bins")
    plt.ylabel("r between reference and experiment")
    plt.title(title)
    name = '_'.join(title.lower().split()).replace(' ', '_')
    plt.savefig(f"{results_dir}/compare_exp{name}.png")
    print(f"Saved plot {results_dir}/{name}.png")
    plt.show()
    plt.close()

def plot_r_vs_r(df_ref, df_exp, k_array, results_dir, exp_name="yolo"):
    """
    Compare the reference simlation to experiments in terms of correlation coefficients.
    Input: df_ref (pandas dataframe) with the evaluation metrics for the reference simulation (x-axis).
           df_ref (pandas dataframe) with the evaluation metrics for the experiment (y-axis).
           k_array (numpy array) with the k values.
           output_dir (str) with the path to the output directory.
           exp_name (str) with the name of the experiment.
    Output: plot
    """
    for s in ['mix', 'ha', 'oiii']:
        for i in range(len(k_array)-1): 
            k = str(int(k_array[i]))
            # First sort the values in the dataframe from the reference simulation
            row = "r_"+s+"_"+k
            sorted_ref =df_ref.sort_values(by=row, ascending=True)
            sorted_exp = df_exp.loc[sorted_ref.index]
            vmax = max(sorted_ref[row].max(), sorted_exp[row].max())
            vmin = min(sorted_ref[row].min(), sorted_exp[row].min())

            plt.figure(figsize=(10, 6))
            plt.plot(sorted_ref[row], df_exp[row], label=row, marker='o', linestyle='None')
            plt.legend()
            plt.xlabel("reference r between true and fake at k= "+str(k))
            plt.ylabel("experiment r between true and fake at k= "+str(k))
            plt.xlim(vmin, vmax)
            plt.ylim(vmin, vmax)
            name = exp_name+"_r_vs_r_"+row
            plt.title(name)
            plt.savefig(f"{results_dir}/{name}.png")
            print(f"Saved plot {results_dir}/{name}.png")
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

def plot_r_single_sample(data, suffix, log_bins=True):
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

    r_mix , k_array = compute_r(m, df_fake["rec"].values[0], log_bins=log_bins)
    r_ha , _ = compute_r(a, df_fake["fakeA"].values[0], log_bins=log_bins)
    r_oiii , _ = compute_r(b, df_fake["fakeB"].values[0], log_bins=log_bins)
    
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
    #plt.savefig(f"{results_dir}/compare_exp{suffix}.png")
    #print(f"Saved plot {results_dir}/{suffix}.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    base_output_dir = "../output/" # Meanwhile I have my own output directory with GAN results
    names = ['test', 'xai_exp_occlusion' ]
    results_dir = "../output/xai_occlusion_results/"
    nrun = 100
    nindex = 1
    suffix_list = [ "run{:d}_index{:d}".format(i, j) for i in range(nrun) for j in range(nindex) ]
    ref_dir = os.path.join(base_output_dir, names[0])
    occ_dir = os.path.join(base_output_dir, names[1])
    suffix = f"run1_index0"

    for i in range(16):
        df = read_occ_data(occ_dir, i, suffix=suffix)
        plot_occluded_map(df, results_dir, i, exp_name=names[1], suffix=suffix)

    exit()
    base_output_dir = "../output/" # Meanwhile I have my own output directory with GAN results
    names = ['test', 'xai_exp_only_using_ha', 'xai_exp_only_using_oiii' ]
    results_dir = "../output/xai_results/"
    print("Reading data from ", base_output_dir)
    print("Writing results to ", results_dir)

    # Check if the output directories exists
    if not os.path.exists(base_output_dir):
        print(f"Output directory {base_output_dir} does not exist.")
        print("The ouput directory stores the output of the GAN needed as input for XAI.")
        exit()
    if not os.path.exists(results_dir):
        # Actual output directory to store XAI related results
        os.makedirs(results_dir)
        

    nrun = 100
    nindex = 1
    moments = ["mean", "std", "max", "sum"]
    error_metrics = ["l1", "rmse", "d_sum"]
    container = {}
    for d in names:
        output_dir = os.path.join(base_output_dir, d)
        if not os.path.exists(output_dir):
            print(f"Output directory {output_dir} does not exist.")
            print("The ouput directory stores the output of the GAN needed as input for XAI.")
            exit()
        df, k_array = create_dataframe(output_dir, nrun, nindex)
        for m in moments:
            plot_true_vs_k(df, k_array, results_dir, moment=m, exp_name=d)
        for e in error_metrics:
            plot_k_vs_error(df, k_array, results_dir, error=e, exp_name=d)
        container[d] = df

    df_ref = container[names[0]]
    df_ha = container[names[1]]
    df_oiii = container[names[2]]
    plot_r_vs_r(df_ref, df_ha, k_array, results_dir, exp_name=names[1])
    plot_r_vs_r(df_ref, df_oiii, k_array, results_dir, exp_name=names[2])
    #pdb.set_trace()
    print("DONE")
    