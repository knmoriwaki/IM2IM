import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MyModel

from utils import *

parser = argparse.ArgumentParser(description="")
parser.add_argument("--isTrain", dest="isTrain", action='store_true', help="train or test")
parser.add_argument('--gpu_ids', dest="gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--load_iter', dest="load_iter", type=int, default='0', help='If load_iter > 0, the code will load models by iter_[load_iter]. If load_iter = -1, the code will load the latest model')

parser.add_argument("--data_dir", dest="data_dir", type=str, default="./training_data", help="Root directory of training dataset")
parser.add_argument("--val_dir", dest="val_dir", type=str, default="./val_data", help="Root directory of validation dataset")
parser.add_argument("--test_dir", dest="test_dir", type=str, default="./test_data", help="Root directory of test dataset")
parser.add_argument('--output_dir', dest="output_dir", type=str, default='./output', help='all the outputs and models are saved here')
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

parser.add_argument("--d_model", dest="d_model", type=int, default=64, help="the number of expected features in the encoder/decoder inputs")
parser.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="dropout rate")
#parser.add_argument("--nhead", dest="nhead", type=int, default=8, help="the number of heads in the multiheadattention models")
#parser.add_argument("--num_layers", dest="num_layers", type=int, default=6, help="the number of sub-encoder-layers in the encoder")
#parser.add_argument("--dim_feedforward", dest="dim_feedforward", type=int, default=1024, help="the dimension of the feedforward network model")
#parser.add_argument("--activation", dest="activation", type=str, default="relu", help="the activation function")
#parser.add_argument("--last_activation", dest="last_activation", type=str, default="tanh", help="the activation function")

# training parameters #
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size")
parser.add_argument("--n_epochs", dest="n_epochs", type=int, default=100, help="training epoch")
parser.add_argument('--n_epochs_decay', dest="n_epochs_decay", type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument("--epoch_count", dest="epoch_count", type=int, default=1, help="the starting epoch count")
parser.add_argument("--lr", dest="lr", type=float, default=0.0002, help="learning rate")
parser.add_argument('--lr_policy', dest="lr_policy", type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', dest="lr_decay_iters", type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument("--beta1", dest="beta1", type=float, default=0.5, help="beta1 of Adam optimizer")

parser.add_argument("--lambda_L1", dest="lambda_L1", type=float, default=100.0, help="weight for L1 loss in GAN")

parser.add_argument("--print_freq", dest="print_freq", type=int, default=100, help="frequency of showing training results on console")
parser.add_argument("--save_latest_freq", dest="save_latest_freq", type=int, default=5000, help="frequency of saving the latest results")
parser.add_argument("--save_image_freq", dest="save_image_freq", type=int, default=10000, help="frequency of saving the generated image")
parser.add_argument("--save_image_irun", dest="save_image_irun", type=int, default=-1, help="id of saved image during training")
args = parser.parse_args()

def main():

    device = my_init(seed=0, gpu_ids="0")
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
    target = data_list[1] ## train to output oiii data
    print("   Time Taken: {:.0f} sec".format(time.time() - start_time)) 
    return source, target
    
def train(device):

    ### load data ###
    prefix_list = [ "rea{:d}/run{:d}_index{:d}".format(irea, i, j) for irea in range(1) for i in range(args.nrun) for j in range(args.nindex) ]
    source, target = load_data(args.data_dir, prefix_list, device=None)
    dataset = MyDataset(source, target)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("# batch_size: ", args.batch_size)
    print("# source data: ", np.shape(source))
    print("# target data: ", np.shape(target))

    ### define model ###
    model = MyModel(args)
    model.setup(args, verbose=True) #set verbose=True to show the model architecture

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
                    model.set_input([src_ref, tgt_ref])
                    fname = "{}/{}_iter_{:d}.fits".format(args.output_dir, prefix_list[args.save_image_irun], total_iters)
                    model.save_test_image(args, fname, overwrite=True) 
                else:
                    fname = "{}/iter_{:d}.fits".format(args.output_dir, total_iters)
                    visuals = model.get_current_visuals()
                    save_image(visuals["fake_B"][0], fname, args.norm, overwrite=True)
                print("# save {}".format(fname))
            
            del src, tgt
            torch.cuda.empty_cache()

        print('# End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs + args.n_epochs_decay, time.time() - epoch_start_time))

    print('# saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
    model.save_networks('latest')

def test(device):

    ### load data ###
    prefix_list = [ "run{:d}_index{:d}".format(i, j) for i in range(args.nrun) for j in range(args.nindex) ]
    source, target = load_data(args.test_dir, prefix_list, device=device)

    ### load model ###
    model = MyModel(args)
    model.setup(args, verbose=False)

    model.eval()

    ### test ###
    for i, (src, tgt, p) in enumerate(zip(source, target, prefix_list)):
        src = torch.unsqueeze(src, 0)
        tgt = torch.unsqueeze(tgt, 0)
        if args.load_iter > 0: suf += "iter{:d}".format(args.load_iter)
        model.set_input([src,tgt])

        fname = "{}/test/gen_{}.fits".format(args.output_dir, p) 
        model.save_test_image(args, fname, overwrite=True)
        print("# save {}".format(fname))

if __name__ == "__main__":
    main()
