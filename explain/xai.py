import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.io import fits
from correlation_coefficient import compute_r
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

def read_data(output_dir, xai_exp=None, suffix=f"run0_index0",ldict=False):
    f_realA = f"/mnt/data_cat4/moriwaki/IM2IM/val_data/{suffix}_z1.3_ha.fits"
    f_realB = f"/mnt/data_cat4/moriwaki/IM2IM//val_data/{suffix}_z2.0_oiii.fits"
    f_fakeA = f"{output_dir}/gen_{suffix}_0.fits"
    f_fakeB = f"{output_dir}/gen_{suffix}_1.fits"

    f_list = [ f_realA, f_realB, f_fakeA, f_fakeB ]
    data = [ fits.open( f )[0].data for f in f_list ]
    label_list = ["obs", "trueHa", "trueOIII", "rec", "fakeHa", "fakeOIII"]    
    
    # Check if xai_exp is not none
    if xai_exp == 'ha':
        print(xai_exp)
        data[1] = data[1]*0.0
        data = [ data[0]+data[1], data[0], data[1], data[2]+data[3], data[2], data[3] ]
    elif xai_exp == 'oiii':
        print(xai_exp)
        data[0] = data[0]*0.0
        data = [ data[0]+data[1], data[0], data[1], data[2]+data[3], data[2], data[3] ]
    elif xai_exp == 'random':
        print(xai_exp)
        data[0] = np.random.random(np.shape(data[0]))*0.1
        data[1] = np.random.random(np.shape(data[0]))*0.1
        data = [ data[0]+data[1], data[0], data[1], data[2]+data[3], data[2], data[3] ]
    else:
        data = [ data[0]+data[1], data[0], data[1], data[2]+data[3], data[2], data[3] ]
        
    if ldict:
        data = { l:d for l, d in zip(label_list, data) }
        
    return data

def read_occ_data(output_dir, n_occ, suffix=f"run0_index0"):
    """
    Read the original inputs, but also the perturbed inputs as 
    perturbed during testing and the generated images. 
    Returns a pandas dataframe for a single sample.
    """
    
    f_realA = f"/mnt/data_cat4/moriwaki/IM2IM/val_data/{suffix}_z1.3_ha.fits"
    f_realB = f"/mnt/data_cat4/moriwaki/IM2IM//val_data/{suffix}_z2.0_oiii.fits"
    f_fakeA = f"{output_dir}/gen_{suffix}_occluded{n_occ}_0.fits"
    f_fakeB = f"{output_dir}/gen_{suffix}_occluded{n_occ}_1.fits"
    f_pertA = f"{output_dir}/occluded_input_{suffix}_occluded{n_occ}_target_0.fits"
    f_pertB = f"{output_dir}/occluded_input_{suffix}_occluded{n_occ}_target_1.fits"
    f_pertC = f"{output_dir}/occluded_input_{suffix}_occluded{n_occ}_source.fits"
    
    # Construct lists for opening the files
    f_real = [ f_realA, f_realB ]
    f_fake = [ f_fakeA, f_fakeB ]
    f_pert = [ f_pertC, f_pertA, f_pertB ]
    # Open the files and construct a data list and corresponding keys
    raw_r  = [ fits.open( f )[0].data for f in f_real ]
    data_r = [ raw_r[0]+raw_r[1], raw_r[0], raw_r[1] ]
    keys_r = ['obs', 'realA', 'realB']
    raw_f  = [ fits.open( f )[0].data for f in f_fake ]
    data_f = [ raw_f[0]+raw_f[1], raw_f[0], raw_f[1] ]
    keys_f = ['rec', 'fakeA', 'fakeB']
    data_p = [ fits.open( f )[0].data for f in f_pert ]
    keys_p = ['p_s', 'p_tA', 'p_tB'] # p_s: perturbed source, p_t: perturbed target
    # Create dictionaries with keys and data
    dict_r = dict(zip(keys_r, data_r))
    dict_f = dict(zip(keys_f, data_f))
    dict_p = dict(zip(keys_p, data_p))
    # Convert to pandas dataframes
    df_r = pd.DataFrame.from_dict({k: [v] for k, v in dict_r.items()})
    df_f = pd.DataFrame.from_dict({k: [v] for k, v in dict_f.items()})
    df_p = pd.DataFrame.from_dict({k: [v] for k, v in dict_p.items()})
    # Concatenate all into one dataframe
    df = pd.concat([df_r, df_p, df_f], axis=1)
    
    return df

def plot_true_fake_maps(data, results_dir):
    # reproduced map
    label_list = ["observed", "true A", "true B", "observed (rec)", "reconstructed A", "reconstructed B"]
    vmin = 0
    #vmax = np.max(data[0])
    vmax = 9.0e-08

    fig, axs = plt.subplots(2,3)

    for i, (d, l) in enumerate(zip(data, label_list)):    
        ax = axs[int(i/3)][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(l)
        im = ax.imshow(d, interpolation="none", vmin=vmin, vmax=vmax)
    
    filename ="test_image.png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path)
    plt.show()

def plot_occluded_map(df, results_dir, n_occ, exp_name='occ', suffix=f"run0_index0"):
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
        
    filename =f"{exp_name}_{suffix}_occluded{n_occ}_image.png"    
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path) 
    plt.show()
    plt.close()

def plot_r_occ_sample(ref_dir, occ_dir, results_dir, n_occ, suffix, nbins=20, log_bins=True):
    """
    Inputs: df (dataframe holding the real, perturbed (occluded) and generated images)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot occluded 
    l_mix = []
    l_ha = []
    l_oiii = []
    for i in range(int(n_occ)):
        df = read_occ_data(occ_dir, i, suffix=suffix)
        r_mix , k_array = compute_r(df["p_s"].values[0], df["rec"].values[0], nbins=nbins, log_bins=log_bins)
        r_ha , _ = compute_r(df["p_tA"].values[0], df["fakeA"].values[0], nbins=nbins, log_bins=log_bins)
        r_oiii , _ = compute_r(df["p_tB"].values[0], df["fakeB"].values[0], nbins=nbins, log_bins=log_bins)
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
    data = read_data(ref_dir, suffix=suffix, ldict=True)
    r_mix , _ = compute_r(data["obs"], data["rec"], log_bins=log_bins)
    r_ha , _ = compute_r(data["trueHa"], data["fakeHa"], log_bins=log_bins)
    r_oiii , _ = compute_r(data["trueOIII"], data["fakeOIII"], log_bins=log_bins)
    
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

def calc_importance(ref_dir, occ_dir, n_occ, suffix, nbins=20, log_bins=True):
    """
    I want to assign an importance defied by the difference between 
    reference(r_(true-fake)) and experiment(r_(perturbed_true-fake)) to each patch in the occlusion
    """ 
    
    # Read reference data
    data = read_data(ref_dir, suffix=suffix, ldict=True)
    ref_mix , _ = compute_r(data["obs"], data["rec"], log_bins=log_bins)
    ref_ha , _ = compute_r(data["trueHa"], data["fakeHa"], log_bins=log_bins)
    ref_oiii , _ = compute_r(data["trueOIII"], data["fakeOIII"], log_bins=log_bins)

    im_size = 256

    l_mix = []
    l_ha = []
    l_oiii = []
    for i in range(n_occ):
        # Read occluded data
        df = read_occ_data(occ_dir, i, suffix=suffix)
        # Calculate correlation coefficients
        r_mix , _ = compute_r(df["p_s"].values[0], df["rec"].values[0], nbins=nbins, log_bins=log_bins)
        r_ha , _ = compute_r(df["p_tA"].values[0], df["fakeA"].values[0], nbins=nbins, log_bins=log_bins)
        r_oiii , _ = compute_r(df["p_tB"].values[0], df["fakeB"].values[0], nbins=nbins, log_bins=log_bins)
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
    occlusion_size = int(np.sqrt(im_size*im_size/n_occ))
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
        #container[d] = df

    #pdb.set_trace()
    print("DONE")
    