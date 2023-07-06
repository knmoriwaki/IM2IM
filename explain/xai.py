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

def read_occ_data(output_dir, n_occluded, suffix=f"run0_index0"):
    """
    Inputs:
        output_dir: directory where the generated and occluded fits files are stored.
        n_occluded: number of occluded patches 
        xai_exp   : optional maybe can be removed?
        suffix    : this function only uses a single sample so the suffix identifies this sample
        ldict     : ?? can be removed? how to do it?
    """
    
    f_realA = f"/mnt/data_cat4/moriwaki/IM2IM/val_data/{suffix}_z1.3_ha.fits"
    f_realB = f"/mnt/data_cat4/moriwaki/IM2IM//val_data/{suffix}_z2.0_oiii.fits"
    f_list = [ f_realA, f_realB ]
    for i_occ in range(n_occluded):
        f_occ_real = f"{output_dir}/occluded_input_{suffix}_occluded{i_occ}_source.fits" #you need to know how many there are!
        f_list.append(f_occ_real)
    for i_occ in range(n_occluded):
        f_fakeA = f"{output_dir}/gen_{suffix}_occluded{i_occ}_0.fits"
        f_list.append(f_fakeA)
        f_fakeB = f"{output_dir}/gen_{suffix}_occluded{i_occ}_1.fits"
        f_list.append(f_fakeB)
        
    data = [ fits.open( f )[0].data for f in f_list ] 
    
    ground_truth = [ data[0]+data[1], data[0], data[1]]
    occ_truth = data[2:n_occluded+2] # +2 because the first two entries are the true/real values followed by n_occluded occluded images
    assert len(occ_truth) == n_occluded
    fake = data[n_occluded+2:] # followed by 2*n_occluded generated images
    assert len(fake) == 2*n_occluded
    reshaped_fake = [fake[i:i+2] for i in range(0, len(fake), 2)]
        
    return [ground_truth, occ_truth, reshaped_fake], f_list

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

def plot_occ_maps(results_dir, data, suffix, patch=None):
    """
    Plots for a single occluded example the maps for sanity check.
    """
    p_data = []
    p_data = data[0] # Initialize using the truth
    occ_truth = data[1] # Has all the occluded sources
    occ_fake  = data[2] # Has all the generated fakes based on the respective occluded source
    
    # add the occluded source input such at the generated fakeA and fakeB
    # First choose a random patch if the user did not specify a certain patch
    if patch != None:
        patch = int(patch)
    else:
        # pick random occluded patch for visualization
        patch = random.randint(0, len(occ_truth))

    print(patch)
    p_data.append(occ_truth[patch])
    p_data.append(occ_fake[patch][0]) # Halpha
    p_data.append(occ_fake[patch][1]) # OIII
    
    print(len(p_data))
    #assert len(p_data) == 6

    label_list = ["observed", "true A", "true B", "occluded input", "reconstructed A", "reconstructed B"]
    
    vmin = 0
    vmax = np.max(p_data[0])

    fig, axs = plt.subplots(2,3)
    for i, (d, l) in enumerate(zip(p_data, label_list)):    
        ax = axs[int(i/3)][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(l)
        ax.imshow(d, interpolation="none", vmin=vmin, vmax=vmax)
    
    filename = f"occluded_result_{suffix}_patch{patch}.png"  
    save_path = os.path.join(results_dir, filename)    
    plt.savefig(save_path)
    plt.show()
    plt.close()
    del p_data, occ_truth, occ_fake, data



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

    data, f_list = read_occ_data(occ_dir, 16, suffix=suffix)
    #plot_occ_maps(results_dir, data, suffix, patch=1)
    for p in range(16):
        plot_occ_maps(results_dir, data, suffix, patch=p)

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
    