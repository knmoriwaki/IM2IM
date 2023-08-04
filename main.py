import sys
import argparse
import time
import json
import pdb
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from model import MyModel

from utils import *
from explain.explainability_functions import xai_load_data, occlusion_load_data

#torch.autograd.set_detect_anomaly(True) ## For debug -- detect places where backpropagation doesn't work properly

parser = argparse.ArgumentParser(description="")
parser.add_argument("--isTrain", dest="isTrain", action='store_true', help="train or test")
parser.add_argument("--isXAI", dest="isXAI", action='store_true', help="Run XAI Experiment")
parser.add_argument('--gpu_ids', dest="gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--load_iter', dest="load_iter", type=int, default='0', help='If load_iter > 0, the code will load models by iter_[load_iter]. If load_iter = -1, the code will load the latest model')

parser.add_argument("--data_dir", dest="data_dir", type=str, default="./training_data", help="Root directory of training dataset")
parser.add_argument("--val_dir", dest="val_dir", type=str, default="./val_data", help="Root directory of validation dataset")
parser.add_argument("--test_dir", dest="test_dir", type=str, default="./test_data", help="Root directory of test dataset")
parser.add_argument('--output_dir', dest="output_dir", type=str, default='./output', help='all the outputs and models are saved here')
parser.add_argument('--results_dir', dest="results_dir", type=str, default='./output', help='optional, inference outputs are saved here')
parser.add_argument('--name', dest="name", type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

parser.add_argument("--norm", dest="norm", type=float, default=1.0, help="Normalization")
parser.add_argument("--nrun", dest="nrun", type=int, default=100, help="number of realizations")
parser.add_argument("--nindex", dest="nindex", type=int, default=1, help="number of indexes for each realization")

# model parameters #
parser.add_argument("--model", dest="model", type=str, default="pix2pix", help="model name")
parser.add_argument("--input_nc", dest="input_nc", type=int, default=1, help="the number of input channels")
parser.add_argument("--output_nc", dest="output_nc", type=int, default=1, help="the number of output channels")
parser.add_argument("--input_dim", dest="input_dim", type=int, default=256, help="the number of input pixels along x/y axis")
parser.add_argument("--output_dim", dest="output_dim", type=int, default=256, help="the number of output pixels along x/y axis")

parser.add_argument("--hidden_dim_G", dest="hidden_dim_G", type=int, default=64, help="the number of expected features in the first layer of the generator")
parser.add_argument("--hidden_dim_D", dest="hidden_dim_D", type=int, default=64, help="the number of expected features in the first layer of the discriminator")
parser.add_argument("--nlayer_G", dest="nlayer_G", type=int, default=8, help="the number of layers in the generator")
parser.add_argument("--nlayer_D", dest="nlayer_D", type=int, default=4, help="the number of layers in the discriminator")
parser.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="dropout rate")

# training parameters #
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size")
parser.add_argument("--n_epochs", dest="n_epochs", type=int, default=100, help="training epoch")
parser.add_argument('--n_epochs_decay', dest="n_epochs_decay", type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument("--epoch_count", dest="epoch_count", type=int, default=1, help="the starting epoch count")
parser.add_argument("--lr", dest="lr", type=float, default=0.0002, help="learning rate")
parser.add_argument('--lr_policy', dest="lr_policy", type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', dest="lr_decay_iters", type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument("--beta1", dest="beta1", type=float, default=0.5, help="beta1 of Adam optimizer")

parser.add_argument("--lambda_L1", dest="lambda_L1", type=float, default=10.0, help="weight for L1 loss in GAN")
parser.add_argument("--gan_mode", dest="gan_mode", default="vanilla", help="GAN loss -- vanilla, wgan, wgangp, lsgan")

parser.add_argument("--print_freq", dest="print_freq", type=int, default=100, help="frequency of showing training results on console")
parser.add_argument("--save_latest_freq", dest="save_latest_freq", type=int, default=5000, help="frequency of saving the latest results")
parser.add_argument("--save_image_freq", dest="save_image_freq", type=int, default=10000, help="frequency of saving the generated image")
parser.add_argument("--save_image_irun", dest="save_image_irun", type=int, default=-1, help="id of saved image during training")
parser.add_argument("--data_aug", dest="data_aug", type=float, help="Fraction images robust data augmentation between 0. and 1.")

# XAI parameters #
parser.add_argument("--xai_exp",  choices=['ha', 'oiii', 'random', 'random_ha', 'random_oiii', 'faint_ha', 'occlusion'], type=str, default=None, 
                    help="The user needs to choose which XAI expierment to perform.")
parser.add_argument("--occlusion_size", type=int, default=64, help="Occlusion window size for occlusion sensitivity")
parser.add_argument("--occlusion_stride", type=int, default=32, help="Occlusion stride for occlusion sensitivity")
parser.add_argument("--occlusion_sample", type=int, help="Sample id for occlusion sensitivity")

args = parser.parse_args()

def main():
    device = my_init(seed=0, gpu_ids=args.gpu_ids)

    if args.model == "pix2pix_2" and args.output_nc != 2:
        print("Warning: inconsinstent output_nc. Use output_nc = 2")
        args.output_nc = 2

    if args.isTrain:
        with open("{}/params.json".format(args.output_dir), mode="a") as f:
            json.dump(args.__dict__, f)
        train(device)
    else:
        test(device)


def load_data(path, prefix_list, device="cuda:0"):

    start_time = time.time()
    print("# loading data from {}".format(path), end=" ")
    data_list = []
    for label in [ "z1.3_ha", "z2.0_oiii" ]:
        fnames = [ "{}/{}_{}.fits".format(path, p, label) for p in prefix_list ]
        data = load_fits_image(fnames, norm=args.norm, device=device)
        data_list.append(data)
    source = data_list[0] + data_list[1]
    target1 = data_list[0] 
    target2 = data_list[1] 
    print("   Time Taken: {:.0f} sec".format(time.time() - start_time)) 

    if args.model == "pix2pix_2":
        target = torch.cat((target1, target2), 1) #(N, 2, Npix, Npix)
    else:
        target = target2
    
    target = torch.clamp(target, min=-1.0, max=1.0)
    return source, target

def load_augmented_data(path, prefix_list, p_aug=0.1, device="cuda:0"):
    """
    Load augmented data for training. First try only augment using OIII.
    """

    start_time = time.time()
    print("# loading data from {}".format(path), end=" ")

    # Random selection of p% of the data for augmentation
    # Calculate the number of elements n_aug you want to select
    n_aug = int(len(prefix_list) * p_aug)
    # Randomly select 10% (90 elements) from the list without replacement
    aug_prefix_list = random.sample(prefix_list, n_aug)


    data_list = []
    for label in [ "z1.3_ha", "z2.0_oiii" ]:
        fnames = [ "{}/{}_{}.fits".format(path, p, label) for p in prefix_list ]
        data = load_fits_image(fnames, norm=args.norm, device=device)

        if 0. < p_aug < 1.:
            #random sample the robustness factor between 0.8 and 1.2
            robustness_factor = 0.4 * torch.rand(n_aug) + 0.8
            robustness_factor = robustness_factor.view(n_aug, 1, 1, 1)     
            fnames = [ "{}/{}_{}.fits".format(path, p, label) for p in aug_prefix_list ]
            aug_data = load_fits_image(fnames, norm=args.norm, device=device)
            aug_data = aug_data*robustness_factor
            print("Augmenting", label)
            print(aug_data.size())
            print("factor", robustness_factor[1,:,:,:])
            data = torch.cat((data, aug_data), 0)
        elif p_aug == 1.:
            robustness_factor = 0.4 * torch.rand(n_aug) + 0.8
            robustness_factor = robustness_factor.view(n_aug, 1, 1, 1)    
            print("Augmenting", label)
            print("factor", robustness_factor[1,:,:,:])
            data = data*robustness_factor
        
        data_list.append(data)
    
    source = data_list[0] + data_list[1]
    target1 = data_list[0] 
    target2 = data_list[1] 
    print("   Time Taken: {:.0f} sec".format(time.time() - start_time)) 

    if args.model == "pix2pix_2":
        target = torch.cat((target1, target2), 1) #(N, 2, Npix, Npix)
    else:
        target = target2
    
    target = torch.clamp(target, min=-1.0, max=1.0)

    return source, target
    
def train(device):
 
    ### define model ###
    model = MyModel(args)
    model.setup(args, verbose=True) #set verbose=True to show the model architecture
    
    summary(model.netG, input_size=(args.batch_size, args.input_nc, args.input_dim, args.input_dim), col_names=["output_size", "num_params"], device=device)
    summary(model.netD, input_size=(args.batch_size, args.input_nc+args.output_nc, args.input_dim, args.input_dim), col_names=["output_size", "num_params"], device=device)
    

    ### load data ###
    prefix_list = [ "rea{:d}/run{:d}_index{:d}".format(irea, i, j) for irea in range(3) for i in range(args.nrun) for j in range(args.nindex) ]
    if args.data_aug:
        source, target = load_augmented_data(args.data_dir, prefix_list, args.data_aug, device=None)
    else:
        source, target = load_data(args.data_dir, prefix_list, device=None)
    dataset = MyDataset(source, target)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if np.shape(source)[1] != args.input_nc:
        print("Error: inconsistent input_nc")
        sys.exit(1)

    if np.shape(target)[1] != args.output_nc:
        print("Error: inconsistent output_nc")
        sys.exit(1)

    print("# batch_size: ", args.batch_size)
    print("# source data: ", np.shape(source))
    print("# target data: ", np.shape(target))

    ### training ###
    niters_per_epoch = len(train_loader)
    total_iters = args.load_iter if args.load_iter > 0 else 0
    start_time = time.time()
    for epoch in range(args.epoch_count, args.n_epochs + args.n_epochs_decay + 1):
        epoch_start_time = time.time()
        if epoch != args.epoch_count:
            model.update_learning_rate()

        for i, (src, tgt) in enumerate(train_loader):
            total_iters += 1

            model.set_input([src,tgt])
            model.optimize_parameters() # calculate loss functions, get gradients, update network weights
            if total_iters == 1:
                losses = model.get_current_losses()
                print_loss_names(args, losses)
            if total_iters % args.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_now = time.time() - start_time
                epoch_now = float( total_iters ) / float( niters_per_epoch )
                print_current_losses(args, total_iters, total_iters*args.batch_size, epoch_now, losses, t_now)

            if total_iters % args.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('# saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('iter_{:d}'.format(total_iters))
                model.save_networks('latest')

            if total_iters % args.save_image_freq == 0:
                if args.save_image_irun >= 0:
                    src_ref = torch.unsqueeze(source[args.save_image_irun], 0)
                    tgt_ref = torch.unsqueeze(target[args.save_image_irun], 0)
                    fid = "{}/{}_iter_{:d}".format(args.output_dir, prefix_list[args.save_image_irun], total_iters)
                    model.save_test_image(args, fid, overwrite=True) 
                else:
                    fid = "{}/iter_{:d}".format(args.output_dir, total_iters)
                    visuals = model.get_current_visuals()
                    for iout in range(args.output_nc):
                        fname = "{}_{:d}.fits".format(fid, iout)
                        save_image(visuals["fake_B"][0][iout], fname, args.norm, overwrite=True)
                print("# save {}_*.fits".format(fid))
            
            del src, tgt
            torch.cuda.empty_cache()

        print('# End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs + args.n_epochs_decay, time.time() - epoch_start_time))

    print('# saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
    model.save_networks('latest')

def test(device):
    start_time = time.time()
    if args.isXAI:
        exp_dir = "xai_exp_" + args.xai_exp
    else:
        exp_dir = "test"

    # Make sure the subdirectory exists, if not create it
    res_dir = "{}/{}".format(args.results_dir, exp_dir)
    if not os.path.exists(res_dir):
        print("# create {}".format(res_dir))
        os.makedirs(res_dir)


    ### load data ###
    prefix_list = [ "run{:d}_index{:d}".format(i, j) for i in range(args.nrun) for j in range(args.nindex) ]
    if args.isXAI:
        if args.xai_exp in ["ha", "oiii", "random", "random_ha", "random_oiii", "faint_ha"]:
            source, target = xai_load_data(args, prefix_list, device=device)
        elif args.xai_exp=="occlusion":
                if args.occlusion_sample is not None:
                    #pdb.set_trace()
                    prefix_list = [prefix_list[args.occlusion_sample]]
                    source, target, n_occluded = occlusion_load_data(args, prefix_list, device=device)
                    print("# Occluding {}x{} image patches. For sample {}".format(args.occlusion_size, args.occlusion_size, prefix_list))
                    prefix_list = [ "{}_occluded{:d}".format(prefix_list[0], k) for k in range(n_occluded) ]
                else:
                    source, target, n_occluded = occlusion_load_data(args, prefix_list, device=device)
                    print("# Occluding {}x{} image patches for whole test set. This leads to {} times more images in the test set".format(args.occlusion_size, args.occlusion_size, n_occluded))
                    prefix_list = [ "run{:d}_index{:d}_occluded{:d}".format(i, j, k) for i in range(args.nrun) for j in range(args.nindex) for k in range(n_occluded) ]

                
    else:
        source, target = load_data(args.test_dir, prefix_list, device=device)

    ### load model ###
    model = MyModel(args)
    model.setup(args, verbose=False)

    model.eval()

    ### test ###
    for i, (src, tgt, p) in enumerate(zip(source, target, prefix_list)):
        src = torch.unsqueeze(src, 0)
        tgt = torch.unsqueeze(target[i], 0)
        
        model.set_input([src,tgt])

        fid = "{}/gen_{}".format(res_dir, p) 
        model.save_test_image(args, fid, overwrite=True)
        print("# save {}_*.fits".format(fid))
        
        if args.isXAI:
            fid = "{}/perturbed_input_{}".format(res_dir, p) 
            model.save_source_image(args, fid, overwrite=True)
            print("# save {}_*.fits".format(fid))
    print('# End of inference. Time Taken: %d sec' % (time.time() - start_time))

if __name__ == "__main__":
    main()
