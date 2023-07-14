"""
Author: Scarlet Stadtler
Date: July 2023
"""
import os
import pandas as pd
from astropy.io import fits
from save_fits_data import save_fits_data


class XAIDataLoader:
    """
    This class is responsible for loading data from different XAI experiments.
    I decided that it will only support converting the data into pandas dataframes. 
    No lists or dictionaries!
    """
    def __init__(self, output_dir, exp_name, suffix="run71_index0", n_occ=None):
        self.output_dir = output_dir
        self.exp_name = exp_name
        self.exp_dir = os.path.join(self.output_dir, self.exp_name)
        self.suffix = suffix
        if n_occ is not None:
            self.n_occ = n_occ
            self.load_real()
            self.load_single_occluded_fake(self.n_occ)
            self.load_single_occluded_input(self.n_occ)
        else:
            self.load_real()
            self.load_fake()
            self.load_perturbed_inputs()

    def load_real(self):
        """
        Loads a single sample containing the ground truth.
        """
        self.f_realA = f"/mnt/data_cat4/moriwaki/IM2IM/val_data/{self.suffix}_z1.3_ha.fits"
        self.f_realB = f"/mnt/data_cat4/moriwaki/IM2IM//val_data/{self.suffix}_z2.0_oiii.fits"
        f_list = [ self.f_realA, self.f_realB ]
        data = [ fits.open( f )[0].data for f in f_list ]
        data = [ data[0]+data[1], data[0], data[1] ]
        keys = ['obs', 'realA', 'realB']
        tmpd = dict(zip(keys, data))
        self.real = pd.DataFrame.from_dict({k: [v] for k, v in tmpd.items()})
        
    def load_fake(self):
        """
        Loads a single sample containing the generated fake images.
        """
        self.f_fakeA = f"{self.exp_dir}/gen_{self.suffix}_0.fits"
        self.f_fakeB = f"{self.exp_dir}/gen_{self.suffix}_1.fits"
        f_list = [ self.f_fakeA, self.f_fakeB ]
        data = [ fits.open( f )[0].data for f in f_list ]
        data = [ data[0]+data[1], data[0], data[1] ]
        keys = ['rec', 'fakeA', 'fakeB']
        tmpd = dict(zip(keys, data))
        self.fake = pd.DataFrame.from_dict({k: [v] for k, v in tmpd.items()})

    def load_single_occluded_fake(self, n):
        self.f_fakeA = f"{self.exp_dir}/gen_{self.suffix}_occluded{n}_0.fits"
        self.f_fakeB = f"{self.exp_dir}/gen_{self.suffix}_occluded{n}_1.fits"
        f_list = [ self.f_fakeA, self.f_fakeB ]
        try:
            data = [ fits.open( f )[0].data for f in f_list ]
            data = [ data[0]+data[1], data[0], data[1] ]
            keys = ['rec', 'fakeA', 'fakeB']
            tmpd = dict(zip(keys, data))
            self.fake = pd.DataFrame.from_dict({k: [v] for k, v in tmpd.items()})
        except FileNotFoundError as e:
            print(e)
            print("Loading generated fakes failed. check n_occ parameter for consistency.")
            exit()

    def load_perturbed_inputs(self):
        """
        Loads a single sample containing the perturbed inputs. 
        Checks if these exist.
        """
        self.f_pertA = f"{self.exp_dir}/perturbed_input_{self.suffix}_target_0.fits"
        self.f_pertB = f"{self.exp_dir}/perturbed_input_{self.suffix}_target_1.fits"
        self.f_pertC = f"{self.exp_dir}/perturbed_input_{self.suffix}_source.fits"
        f_list = [ self.f_pertC, self.f_pertA, self.f_pertB ] # ! load C first

        try:
            data = [ fits.open( f )[0].data for f in f_list ]
            keys = ['p_s', 'p_tA', 'p_tB'] # p_s: perturbed source, p_t: perturbed target
            tmpd = dict(zip(keys, data))
            self.pert = pd.DataFrame.from_dict({k: [v] for k, v in tmpd.items()})
        except FileNotFoundError as e:
            pass
            #print(e)
            #print("Loading perturbed inputs failed. Check if the files exist.")
            self.pert = None
            
    def load_single_occluded_input(self, n):
        """
        Loads a single sample and single instance of perturbed (occluded) inputs.
        """
        self.f_pertA = f"{self.exp_dir}/perturbed_input_{self.suffix}_occluded{n}_target_0.fits"
        self.f_pertB = f"{self.exp_dir}/perturbed_input_{self.suffix}_occluded{n}_target_1.fits"
        self.f_pertC = f"{self.exp_dir}/perturbed_input_{self.suffix}_occluded{n}_source.fits"
        f_list = [ self.f_pertC, self.f_pertA, self.f_pertB ] # ! load C first
        data = [ fits.open( f )[0].data for f in f_list ]
        keys = ['p_s', 'p_tA', 'p_tB'] # p_s: perturbed source, p_t: perturbed target
        tmpd = dict(zip(keys, data))
        self.pert = pd.DataFrame.from_dict({k: [v] for k, v in tmpd.items()})

    def write(self, image, path, fname, norm=2.0e-7, overwrite=False):
        """Write fits files to disk.
           Args: image to write, path to write to, filename
        """
        size = image.shape
        img = image.reshape(1, 1, size[0], size[1])
        file_name = f"{path}/{fname}.fits"
        save_fits_data(img, file_name, norm=norm, overwrite=overwrite)


# Example usage
if __name__ == "__main__":
    output_dir =  "../output/"
    suffix = "run71_index0"
    exp_name = ["test",  "xai_exp_faint_ha",  "xai_exp_ha", "xai_exp_oiii",  
                "xai_exp_random",  "xai_exp_random_ha",  "xai_exp_random_oiii"]
    #for exp in exp_name:
    #    data_loader = XAIDataLoader(output_dir, exp, suffix)
    #    print(data_loader.pert)
    #    print(data_loader.real)
    #    print(data_loader.fake)

    exp_name = "xai_exp_occlusion"
    data_loader = XAIDataLoader(output_dir, exp_name, suffix, n_occ=2)
    data = data_loader.pert
    save_me = data['p_s'].values[0]
    data_loader.write(save_me, output_dir, "savemetest_input")
