import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from model import MyModel

from utils import *

#torch.autograd.set_detect_anomaly(True) ## For debug -- detect places where backpropagation doesn't work properly

parser = argparse.ArgumentParser(description="")
parser.add_argument("--isTrain", dest="isTrain", action='store_true', help="train or test")
parser.add_argument('--gpu_ids', dest="gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--load_iter', dest="load_iter", type=int, default='0', help='If load_iter > 0, the code will load models by iter_[load_iter]. If load_iter = -1, the code will load the latest model')

parser.add_argument("--data_dir", dest="data_dir", type=str, default="./training_data", help="Root directory of training dataset")
parser.add_argument('--output_dir', dest="output_dir", type=str, default='./output', help='all the outputs and models are saved here')
parser.add_argument('--name', dest="name", type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

parser.add_argument("--norm", dest="norm", type=float, default=1.0, help="Normalization")
parser.add_argument("--ndata", dest="ndata", type=int, default=100, help="number of data (realizations)")
parser.add_argument("--nindex", dest="nindex", type=int, default=1, help="number of indexes for each realization")

# model parameters #
parser.add_argument("--model", dest="model", type=str, default="pix2pix", help="model name")
parser.add_argument("--input_id", dest="input_id", nargs="+", type=str, default="in", help="id(s) of input data")
parser.add_argument("--output_id", dest="output_id", nargs="+", type=str, default="out", help="id(s) of output data")
parser.add_argument("--input_dim", dest="input_dim", type=int, default=256, help="the number of input pixels along x/y axis")

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
args = parser.parse_args()

def main():

    device = my_init(seed=0, gpu_ids=args.gpu_ids)

    if args.isTrain:
        with open("{}/params.json".format(args.output_dir), mode="a") as f:
            json.dump(args.__dict__, f)
        train(device)
    else:
        test(device)

def load_data(base_dir, data_name_list, input_id, output_id, device="cuda:0"):

    start_time = time.time()
    print("# loading data from {}".format(base_dir), end=" ")

    input_list = []
    output_list = []
    for name in data_name_list:
        fname_list = [ "{}/{}_{}.fits".format(base_dir, name, label) for label in input_id ]
        data = load_fits_image(fname_list, norm=args.norm)
        input_list.append(data.unsqueeze(0))

        fname_list = [ "{}/{}_{}.fits".format(base_dir, name, label) for label in output_id ]
        data = load_fits_image(fname_list, norm=args.norm)
        output_list.append(data.unsqueeze(0))
 
    print("   Time Taken: {:.0f} sec".format(time.time() - start_time)) 
    
    source = torch.cat(input_list, dim=0)
    target = torch.cat(output_list, dim=0)

    if device is not None:
        source = source.to(device)
        target = target.to(device)

    return source, target
    ## source: (N, n_feature_in, Npix, Npix)
    ## target: (N, n_feature_out, Npix, Npix)
    
def train(device):

    args.input_id = [args.input_id] if isinstance(args.input_id, str) else args.input_id
    args.output_id = [args.output_id] if isinstance(args.output_id, str) else args.output_id
    n_feature_in = len(args.input_id)
    n_feature_out = len(args.output_id)

    ####################
    ### define model ###
    ####################
    model = MyModel(args)
    model.setup(args, verbose=True) #set verbose=True to show the model architecture
    
    if args.model == "pix2pix":
        summary(model.netG, input_size=(args.batch_size, n_feature_in, args.input_dim, args.input_dim), col_names=["output_size", "num_params"], device=device)
        summary(model.netD, input_size=(args.batch_size, n_feature_in+n_feature_out, args.input_dim, args.input_dim), col_names=["output_size", "num_params"], device=device)
    elif args.model == "vox2vox":
        summary(model.netG, input_size=(args.batch_size, n_feature_in, args.input_dim, args.input_dim, args.input_dim), col_names=["output_size", "num_params"], device=device)
        summary(model.netD, input_size=(args.batch_size, n_feature_in+n_feature_out, args.input_dim, args.input_dim, args.input_dim), col_names=["output_size", "num_params"], device=device)
    
    ####################
    ### load data ######
    ####################
    data_name_list = ["run{:d}_index{:d}".format(i, j) for i in range(args.ndata) for j in range(args.nindex)]
    source, target = load_data(args.data_dir, data_name_list, args.input_id, args.output_id, device=None)
    dataset = MyDataset(source, target)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("# batch_size: ", args.batch_size)
    print("# source data: ", np.shape(source))
    print("# target data: ", np.shape(target))

    ndim = len(source.shape)
    if args.model == "pix2pix":
        if ndim != 4:
            raise ValueError("ndim = {} is not allowed for pix2pix".format(ndim))
    elif args.model == "vox2vox":
        if ndim != 5:
            raise ValueError("ndim = {} is not allowed for vox2vox".format(ndim))


    ####################
    ### training #######
    ####################
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
                    fid = "{}/{}_iter_{:d}".format(args.output_dir, data_name_list[args.save_image_irun], total_iters)
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

    args.input_id = [args.input_id] if isinstance(args.input_id, str) else args.input_id
    args.output_id = [args.output_id] if isinstance(args.output_id, str) else args.output_id

    ### load data ###
    data_name_list = ["run{:d}_index{:d}".format(i, j) for i in range(args.ndata) for j in range(args.nindex)]
    source, target = load_data(args.data_dir, data_name_list, args.input_id, args.output_id, device=device)

    ### load model ###
    model = MyModel(args)
    model.setup(args, verbose=False)
    model.eval()

    ### test ###
    for i, (src, tgt, p) in enumerate(zip(source, target, data_name_list)):
        src = torch.unsqueeze(src, 0)
        tgt = torch.unsqueeze(target[i], 0)
        
        model.set_input([src,tgt])

        fid = "{}/test/gen_{}".format(args.output_dir, p) 
        model.save_test_image(args, fid, overwrite=True)
        print("# save {}_*.fits".format(fid))

if __name__ == "__main__":
    main()
