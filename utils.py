import os
import sys
import numpy as np

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import astropy.io.fits as fits

from tqdm import tqdm

def load_fits_image(fnames, norm=1.0, device="cuda:0"):

    data_list = []
    for fname in fnames:
        img = fits.open(fname)[0].data / norm
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

    
def my_init(seed=0, gpu_ids=-1):
    ## reference: https://pytorch.org/docs/stable/notes/randomness.html

    is_cuda = torch.cuda.is_available()
    if is_cuda and gpu_ids != -1:
        gpu_device_name = 'cuda:{}'.format(gpu_ids[0])
        device = torch.device(gpu_device_name)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("# GPU ({}) is used".format(gpu_device_name))
    else:
        device = torch.device("cpu")
        print("# CPU is used")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8" 

    return device

def print_loss_names(opt, losses):
    message = '#iter iter*batch time'

    for k, v in losses.items():
        message += ' loss_%s' % k

    print(message)  # print the message
    log_name = os.path.join(opt.output_dir, 'loss_log.txt')
    with open(log_name, "w") as log_file:
        log_file.write('%s\n' % message)  # save the messag

def print_current_losses(opt, total_iters, total_iters_b, losses, t):
    message = '%d %d %.3f' % (total_iters, total_iters_b, t)

    for k, v in losses.items():
        message += ' %.3f' % v

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

