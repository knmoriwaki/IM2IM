import time
import pdb
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_fits_image

def xai_load_data(args, prefix_list, device="cuda:0"):
    """ Modification to load data for XAI experiments using only one input instead of the mixed signal.
    """
    path = args.test_dir
    start_time = time.time()
    
    print("# loading data from {}".format(path), end=" ")
    data_list = []
    for label in [ "z1.3_ha", "z2.0_oiii" ]:
        fnames = [ "{}/{}_{}.fits".format(path, p, label) for p in prefix_list ]
        data = load_fits_image(fnames, norm=args.norm, device=device)
        data_list.append(data)
    
    if args.xai_exp == "ha":
        source = data_list[0]
        target1 = data_list[0] 
        target2 = data_list[1]*0.0
        print("Halpha set as source and target. ", torch.mean(source), torch.mean(target1))
        print("This value should be zero: ", torch.mean(target2))
    elif args.xai_exp == "oiii":
        source = data_list[1]
        target1 = data_list[0]*0.0 
        target2 = data_list[1] 
        print("OIII set as source and target. ", torch.mean(source), torch.mean(target2))
        print("This value should be zero: ", torch.mean(target1))
    elif args.xai_exp == "random":
        input1 = data_list[0].to("cpu")
        input2 = data_list[1].to("cpu")
        shuffled1 = input1[:, :, torch.randperm(input1.size()[2]), :]
        shuffled2 = input2[:, :, torch.randperm(input1.size()[2]), :]
        source = shuffled1 + shuffled2
        source = source.to(device)
        target1 = shuffled1.to(device) 
        target2 = shuffled2.to(device)
        print("Shuffled source and target to destroy the structure and keep the statistical distribution.")
    elif args.xai_exp == "random_ha":
        input1 = data_list[0].to("cpu")
        shuffled1 = input1[:, :, torch.randperm(input1.size()[2]), :]
        source = shuffled1.to(device)
        target1 = source 
        target2 = data_list[1]*0.0
        print("Shuffled source and Halpha target, no OIII to destroy the Halpha structure and keep the statistical distribution.")
    elif args.xai_exp == "random_oiii":
        input2 = data_list[1].to("cpu")
        shuffled2 = input2[:, :, torch.randperm(input2.size()[2]), :]
        source = shuffled2.to(device)
        target1 = data_list[0]*0.0
        target2 = source
        print("Shuffled source and Halpha target, no OIII to destroy the Halpha structure and keep the statistical distribution.")
    elif args.xai_exp == "faint_ha":
        brightness_factor = 0.6455
        target1 = data_list[0]*brightness_factor
        target2 = data_list[1]
        source = target1 + target2
        print("Brightness of Halpha scaled down to  ", brightness_factor, " of the original value.")
    else:
        print("Error: no label for the XAI expieriment is specified", args.xai_exp)
        sys.exit(1)

    print("   Time Taken: {:.0f} sec".format(time.time() - start_time)) 

    if args.model == "pix2pix_2":
        target = torch.cat((target1, target2), 1) #(N, 2, Npix, Npix)
    else:
        target = target2
    
    target = torch.clamp(target, min=-1.0, max=1.0)
    return source, target    

def occlusion_load_data(args, prefix_list, device="cuda:0"):
    """ Modification to load data for occlusion experiments.
    source should have this torch.Size([number occluded images, 1, 256, 256])
    target shoud have this torch.Size([number occluded images, 2, 256, 256]) but it is not occluded.
    
    This is meant to be used in main.py with the --occlusion flag.
    Inputs:
        args: arugments passed to the main function
        prefix_list: list of strings with the names of the files to be loaded
        device: device to load the data

    Outputs:
        source: torch tensor with the occluded images
        target: torch tensor with the non-occluded images
    """ 

    path = args.test_dir

    #['run0_index0', 'run1_index0', 'run2_index0', 'run3_index0', 'run4_index0' ] bis 99
    

    start_time = time.time()
    print("# loading data from {}".format(path), end=" ")
    
    data_list_occ_s = []
    data_list_occ_t = []
    for p in prefix_list:
        data_list = []
        for label in [ "z1.3_ha", "z2.0_oiii" ]:
            fnames = ["{}/{}_{}.fits".format(path, p, label)]
            data = load_fits_image(fnames, norm=args.norm, device=device)
            data_list.append(data)
        target1 = data_list[0] 
        target2 = data_list[1]        
        target1 = occlude(target1, args.occlusion_size, args.occlusion_stride, args.occlusion_sample) #(number occluded images, 1, Npix, Npix)
        target2 = occlude(target2, args.occlusion_size, args.occlusion_stride, args.occlusion_sample) #(number occluded images, 1, Npix, Npix)
        source = target1 + target2
        
        if args.model == "pix2pix_2":
            target = torch.cat((target1, target2), 1) #(1, 2, Npix, Npix)
            #n_occ_img = source.size()[0]
            #target = tmp.expand(n_occ_img, -1, -1, -1) #(number occluded images, 1, Npix, Npix)
        else:
            print("Error: Occulsion experiment only works with pix2pix_2 model")
            print("Please set the --model to pix2pix_2")
            print("Exiting...")
            exit()
        
        data_list_occ_s.append(source)
        data_list_occ_t.append(target)
    
    source = torch.cat(data_list_occ_s, 0)
    target = torch.cat(data_list_occ_t, 0)
    print("   Time Taken: {:.0f} sec".format(time.time() - start_time)) 
    n_occluded = source.size()[0]

    target = torch.clamp(target, min=-1.0, max=1.0)
    return source, target, n_occluded


def occlude(target, occlusion_size=64, stride=32, sample=None, masking_type="mean"):
    """ Occludes the source image.
    Inputs:
        source: torch tensor with the source image of size torch.Size([1, 1, 256, 256])
        occlusion_size: size of the occlusion
    Outputs:
        occluded_source: torch tensor with the occluded source image of size torch.Size([n occluded images, 1, 256, 256])
    """
    if masking_type == "mean":
        masking_values = target.mean().item()

    elif masking_type == "zero":
        masking_values = 0.0
    else:
        print("Error: masking type not implemented")
        print("Exiting...")
        exit()

    assert occlusion_size in [8, 16, 32, 64, 128], "Currently supported window sizes are 8, 16, 32, 64"
    occluded_list = []
    rows, cols = target.size()[2:]

    if sample is not None:
        for i in range(0, rows, stride):
            for j in range(0, cols, stride):
                # Copy the target tensor
                tmp = target.clone()
                # Occluding the target with the masking values:
                tmp[0, 0, i:i + occlusion_size, j:j + occlusion_size] = masking_values
                occluded_list.append(tmp)
    else:
        for i in range(0, rows, occlusion_size):
            for j in range(0, cols, occlusion_size):
                # Copy the target tensor
                tmp = target.clone()
                # Occluding the target with the masking values:
                tmp[0, 0, i:i + occlusion_size, j:j + occlusion_size] = masking_values
                occluded_list.append(tmp) 

    occluded_target = torch.cat(occluded_list, dim=0)
    return occluded_target