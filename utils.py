import os
import sys
import math
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import struct

import astropy.io.fits as fits

from tqdm import tqdm

def preprocess(source, target, norm_param_file=None, is_train=True):
    ## Note: target should be within [-1,1] as the last activation layer is Tanh. 
    
    if norm_param_file == None:
        print("# No normalization")
        return source, target

    n_feature_in = source.shape[1]
    n_feature_out = target.shape[1]

    if is_train:
        with open(norm_param_file, 'w') as f:
            for i in range(n_feature_in):
                mean = np.mean(source[:,i])
                std = np.std(source[:,i])
                f.write(f"{mean} {std}\n")

                source[:,i] = (source[:,i] - mean) / std

            if target is not None:
                for i in range(n_feature_out):
                    min_val = np.min(target[:,i])
                    max_val = np.max(target[:,i])
                    f.write(f"{min_val} {max_val}\n")

                    target[:,i] = (target[:,i] - min_val) / (max_val - min_val)
                
                    
        print(f"# Save statistics to {norm_param_file}")

    else:
        with open(norm_param_file, 'r') as f:
            for i, line in enumerate(f):
                val1, val2 = line.split()
                val1 = float(val1)
                val2 = float(val2)
            
                if i < n_feature_in:
                    source[:,i] = (source[:,i] - val1) / val2
                else:
                    if target is not None:
                        ii = i-n_feature_in
                        target[:,ii] = (target[:,ii] - val1) / (val2 - val1)           
                
        print(f"# Load statistics from {norm_param_file}")
    
    return source, target

def denormalize(n_feature_in, source=None, target=None, norm_param_file=None):
    ## Note: target should be within [-1,1] as the last activation layer is Tanh. 
    
    if norm_param_file == None:
        print("# No normalization")
        return source, target

    with open(norm_param_file, 'r') as f:
        for i, line in enumerate(f):
            val1, val2 = line.split()
            val1 = float(val1)
            val2 = float(val2)
        
            if i < n_feature_in:
                if source is not None:
                    source[:,i] = source[:,i] * val2 + val1
            else:
                if target is not None:
                    ii = i-n_feature_in
                    target[:,ii] = (target[:,ii] * (val2 - val1)) + val1
                
    return source, target


def load_Lya_data(data_dir, ndata=10000, npix_patch=[16,16,16], n_feature_in=1, nval=4, norm_param_file=None, is_train=False, device=None):
    
    dm_file = "{}/DM_dens_HnoAGN_realspace_128x128x1024.dat".format(data_dir)
    DM_dens = read_mock3d(dm_file)
    #DM_dens = np.log10( DM_dens )

    if n_feature_in == 2:
        dm_vdisp_file="{}/DM_Vsig_HnoAGN_realspace_128x128x1024.dat".format(data_dir)
        DM_vdisp=read_mock3d(dm_vdisp_file)
        #DM_vdisp = np.log10( DM_vdisp )
        
        DM_input = np.stack([DM_dens, DM_vdisp], axis=0) #(2, dimx, dimy, dimz)
    else:
        DM_input = DM_dens[np.newaxis,:,:,:] #(1, dimx, dimy, dimz)

    flux_file="{}/FLUX_HnoAGN_realspace_128x128x1024.dat".format(data_dir)
    Flux=read_mock3d(flux_file)
    Flux = Flux[np.newaxis,:,:,:] #(1, dimx, dimy, dimz)

    #Flux = np.log10( 1. - Flux )

    ### Remove FGPA prediction
    factor = 0.30
    beta = 0.93
    #Flux -= np.exp(- factor * DM_dens ** beta)

    ### pick up patches of size (npix_patch, npix_patch, npix_patch)
    _, dimx, dimy, dimz = DM_input.shape
    print("# shape: ", DM_input.shape)
    DM_patches = np.zeros((ndata, n_feature_in, npix_patch[0], npix_patch[1], npix_patch[2]))
    Flux_patches = np.zeros((ndata, 1, npix_patch[0], npix_patch[1], npix_patch[2]))
    for ipatch in range(ndata):
        i = np.random.randint(0, dimx - npix_patch[0])
        j = np.random.randint(0, dimy - npix_patch[1])
        k = np.random.randint(0, dimz - npix_patch[2])
        DM_patches[ipatch, :, :, :, :] = DM_input[:, i:i + npix_patch[0], j:j + npix_patch[1], k:k + npix_patch[2]]
        Flux_patches[ipatch, :, :, :, :] = Flux[:, i:i + npix_patch[0], j:j + npix_patch[1], k:k + npix_patch[2]]

    ### normalization
    DM_patches, Flux_patches = preprocess(DM_patches, Flux_patches, norm_param_file=norm_param_file, is_train=is_train)

    ### convert to torch tensor
    DM_patches = torch.from_numpy(DM_patches).float() # (n_feature_in, dimx, dimy, dimz)
    Flux_patches = torch.from_numpy(Flux_patches).float() # (1, dimx, dimy, dimz)

    ### use the last nval data for validation
    if nval > 0:
        DM_train = DM_patches[:-nval]
        Flux_train = Flux_patches[:-nval]
        DM_val = DM_patches[-nval:]
        Flux_val = Flux_patches[-nval:]
    else:
        DM_train = DM_patches
        Flux_train = Flux_patches
        DM_val = None
        Flux_val = None

    ### 
    if device is not None:
        DM_train = DM_train.to(device)
        Flux_train = Flux_train.to(device)
        if nval > 0:
            DM_val = DM_val.to(device)
            Flux_val = Flux_val.to(device)

    return DM_train, Flux_train, DM_val, Flux_val
    # DM_train: tensor of shape (N, n_feature_in, dimx, dimy, dimz)
    # Flux_train: tensor of shape (N, 1, dimx, dimy, dimz)
    # DM_val: tensor of shape (nval, n_feature_in, dimx, dimy, dimz)
    # Flux_val: tensor of shape (nval, 1, dimx, dimy, dimz)


def read_mock3d(mock_file, inlog='false', use_1d=False):
    
    def read_int(data):
        """Read integer (4 bits)"""
        i, = struct.unpack('i', data.read(4))

        return i

    def read_float(data):
        """Read integer (4 bits)"""
        i, = struct.unpack('f', data.read(4))

        return i
        
    """ read mock"""

    data = open(mock_file, "rb")

    ####  READ HEADER
    i0=read_int(data)
    dimx=read_int(data)
    dimy=read_int(data)
    dimz=read_int(data)
    
    if i0==44:  # additional information for DM fields
        xx=read_float(data)
        xx=read_float(data) 
        xx=read_float(data) 
        xx=read_float(data) 
        a=read_float(data) 
        om=read_float(data) 
        ol=read_float(data) 
        hubble=read_float(data); 
        
    i=read_int(data);    
    
    ####  READ SLIDE  
    MAP = np.full((dimy,dimx,dimz), np.nan, dtype = 'float')
    
    Q=1
    for k in range(dimz):
        i=read_int(data);  
        
        for j in range(dimy):
            for i in range(dimx):
                s=struct.unpack('=f', data.read(4))
                if inlog=='false':
                    MAP[i][j][k] = s[0]
                else:   
                    MAP[i][j][k] = math.log10(s[0])
                    if 100.0*(j+1)/1024 > 5*Q:
                        print(5*Q)
                        Q=Q+1

        i=read_int(data);                   
            
    # MAP: (dimy, dimx, dimz)
    if use_1d: 
        MAP = np.reshape(MAP, (dimx*dimy, dimz))
    return MAP

def load_fits_image(fnames, norm=1.0, device="cuda:0"):

    data_list = []
    for fname in fnames:
        with fits.open(fname) as hdul:
            img = hdul[0].data / norm
        size = img.shape
        img = img.reshape(1, 1, size[0], size[1])
        data_list.append(img)
    data_all = np.concatenate(data_list, axis=0)
    data_all = torch.from_numpy( np.array(data_all).astype(np.float32) )
    if device is not None:
        data_all = data_all.to(device)
    return data_all 

def save_image(image, path, norm, overwrite=False):
    img = image.to('cpu').detach().numpy().copy()
    img = img.squeeze()
    img = norm * img
    hdu = fits.PrimaryHDU(img)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=overwrite)

    
def my_init(seed=0, gpu_ids="0"):
    ## reference: https://pytorch.org/docs/stable/notes/randomness.html

    is_cuda = torch.cuda.is_available()
    if is_cuda and gpu_ids != "-1":
        gpu_device_name = 'cuda:{}'.format(gpu_ids[0])
        device = torch.device(gpu_device_name)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  
        torch.backends.cudnn.benchmark = True
        #torch.backends.cuda.matmul.allow_tf32 = False
        #torch.backends.cudnn.allow_tf32 = True ## this makes the speed faster but the precision smaller
        print("# GPU ({}) is used".format(gpu_device_name))
    else:
        device = torch.device("cpu")
        print("# CPU is used")

    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True) # <- not supported for nn.Upsample
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8" 

    return device

def print_loss_names(opt, losses):
    message = '#iter iter*batch epoch time'

    for k, v in losses.items():
        message += ' loss_%s' % k

    print(message)  # print the message
    log_name = os.path.join(opt.output_dir, 'loss_log.txt')
    with open(log_name, "w") as log_file:
        log_file.write('%s\n' % message)  # save the messag

def print_current_losses(opt, total_iters, total_iters_b, epoch, losses, t):
    message = '%d %d %f %.3f' % (total_iters, total_iters_b, epoch, t)

    for k, v in losses.items():
        message += ' %.5f' % v

    print(message)  # print the message
    log_name = os.path.join(opt.output_dir, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the messag

class MyDataset(Dataset):

    def __init__(self, input_list, output_list):

        self.input_list = input_list
        self.output_list = output_list
        self.data_num = len(input_list)

    def __len__(self):
        return self.data_num

    def __getitem__(self, i):
        input = self.input_list[i]
        output = self.output_list[i]

        return input, output

