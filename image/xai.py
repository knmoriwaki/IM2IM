import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from astropy.io import fits
import pdb

name = f"pix2pix_2_bs4_ep1_lambda1000_vanilla"
output_dir = f"/mnt/data_cat4/moriwaki/IM2IM/output/{name}"

def read_data(output_dir, suffix=f"run0_index0", ldict=False):
    f_realA = f"/mnt/data_cat4/moriwaki/IM2IM/val_data/{suffix}_z1.3_ha.fits"
    f_realB = f"/mnt/data_cat4/moriwaki/IM2IM//val_data/{suffix}_z2.0_oiii.fits"
    f_fakeA = f"{output_dir}/test/gen_{suffix}_0.fits"
    f_fakeB = f"{output_dir}/test/gen_{suffix}_1.fits"

    f_list = [ f_realA, f_realB, f_fakeA, f_fakeB ]
    data = [ fits.open( f )[0].data for f in f_list ]
    if ldict:
        label_list = ["obs", "trueHa", "trueOIII", "rec", "fakeHa", "fakeOIII"]
        data = [ data[0]+data[1], data[0], data[1], data[2]+data[3], data[2], data[3] ]
        data = { l:d for l, d in zip(label_list, data) }
    else:
        #label_list = ["observed", "true A", "true B", "observed (rec)", "reconstructed A", "reconstructed B"]
        data = [ data[0]+data[1], data[0], data[1], data[2]+data[3], data[2], data[3] ]
    return data

def plot_true_fake_maps(output_dir):
    # reproduced map
    label_list = ["observed", "true A", "true B", "observed (rec)", "reconstructed A", "reconstructed B"]
    vmin = 0
    vmax = np.max(data[0])

    fig, axs = plt.subplots(2,3)

    for i, (d, l) in enumerate(zip(data, label_list)):    
        ax = axs[int(i/3)][int(i%3)]
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        ax.set_title(l)
        im = ax.imshow(d, interpolation="none", vmin=vmin, vmax=vmax)   
        
    plt.savefig("test_image.png")
    plt.close()

def dbm(data):
    """"Calculate the difference between the mean values. This resembles the L1 Norm,
    but we care about if the difference is positive or negative.
    Input: data (dict) with following labels: ["obs", "trueHa", "trueOIII", "rec", "fakeHa", "fakeOIII"]
    Output: [d, dha, doiii] (list) with difference between mean values for the mixed signal, Halpha and OIII.
    """
    d = np.mean(data["obs"]) - np.mean(data["rec"])
    dha = np.mean(data["trueHa"]) - np.mean(data["fakeHa"])
    doiii = np.mean(data["trueOIII"]) - np.mean(data["fakeOIII"])
    return [d, dha, doiii]

def rmse(data):
    """Calculate the root mean square error between the true and reconstructed maps.
    Input: data (dict) with following labels: ["obs", "trueHa", "trueOIII", "rec", "fakeHa", "fakeOIII"]
    Output: [rmse, rmse_ha, rmse_oiii] (list) with RMSE values for the mixed signal, Halpha and OIII.
    """
    rmse = np.sqrt(np.mean((data["obs"] - data["rec"])**2))
    rmse_ha = np.sqrt(np.mean((data["trueHa"] - data["fakeHa"])**2))
    rmse_oiii = np.sqrt(np.mean((data["trueOIII"] - data["fakeOIII"])**2))
    return [rmse, rmse_ha, rmse_oiii]

if __name__ == "__main__":
    data = read_data(output_dir, ldict=True)
    difference = dbm(data)
    rmse = rmse(data)
    print(difference)
    print(rmse)

    

exit()
# Covariance between true and reconstructed
cov_ha = np.cov(data[0].flatten(), data[2].flatten())
cov_oiii = np.cov(data[1].flatten(), data[3].flatten())
print(cov_ha, cov_oiii)

# Autocorrelation
auto_corr = []
for i in range(4):
    auto = np.cov(data[i].flatten(), data[i].flatten())
    auto_corr.append(auto)
print(auto_corr)
