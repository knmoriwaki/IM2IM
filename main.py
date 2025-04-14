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
parser.add_argument('--gpu_ids', dest="gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--load_iter', dest="load_iter", type=int, default='0', help='If load_iter > 0, the code will load models by iter_[load_iter]. If load_iter = -1, the code will load the latest model')

parser.add_argument("--data_dir", dest="data_dir", type=str, default="./training_data", help="Root directory of training dataset")
parser.add_argument('--output_dir', dest="output_dir", type=str, default='./output', help='all the outputs and models are saved here')
parser.add_argument('--name', dest="name", type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')


# model parameters #
parser.add_argument("--input_dim", dest="input_dim", type=int, default=256, help="the number of input pixels along x/y axis")
parser.add_argument("--output_dim", dest="output_dim", type=int, default=256, help="the number of output pixels along x/y axis")

parser.add_argument("--hidden_dim_G", dest="hidden_dim_G", type=int, default=64, help="the number of expected features in the first layer of the generator")
parser.add_argument("--hidden_dim_D", dest="hidden_dim_D", type=int, default=64, help="the number of expected features in the first layer of the discriminator")
parser.add_argument("--nlayer_G", dest="nlayer_G", type=int, default=4, help="the number of layers in the generator")
parser.add_argument("--nlayer_D", dest="nlayer_D", type=int, default=3, help="the number of layers in the discriminator")
parser.add_argument("--dropout", dest="dropout", type=float, default=0.5, help="dropout rate")
parser.add_argument("--input_noise", dest="input_noise", action="store_true", help="input noise to the generator")


# training parameters #
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size")
parser.add_argument("--n_epochs", dest="n_epochs", type=int, default=100, help="training epoch")
parser.add_argument('--n_epochs_decay', dest="n_epochs_decay", type=int, default=0, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument("--epoch_count", dest="epoch_count", type=int, default=1, help="the starting epoch count")
parser.add_argument("--lr", dest="lr", type=float, default=0.01, help="learning rate")
parser.add_argument('--lr_policy', dest="lr_policy", type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', dest="lr_decay_iters", type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument("--beta1", dest="beta1", type=float, default=0.5, help="beta1 of Adam optimizer")


parser.add_argument("--lambda_L1", dest="lambda_L1", type=float, default=10.0, help="weight for L1 loss in GAN")
parser.add_argument("--lambda_z", dest="lambda_z", type=float, default=0.0, help="weight for input noise in GAN")
parser.add_argument("--gan_mode", dest="gan_mode", default="vanilla", help="GAN loss -- vanilla, wgan, wgangp, lsgan")

parser.add_argument("--print_freq", dest="print_freq", type=int, default=100, help="frequency of showing training results on console")
parser.add_argument("--save_latest_freq", dest="save_latest_freq", type=int, default=5000, help="frequency of saving the latest results")
parser.add_argument("--save_image_freq", dest="save_image_freq", type=int, default=10000, help="frequency of saving the generated image")
parser.add_argument("--save_image_irun", dest="save_image_irun", type=int, default=-1, help="id of saved image during training")
args = parser.parse_args()

def main():

    device = my_init(seed=0, gpu_ids=args.gpu_ids)
    with open("{}/params.json".format(args.output_dir), mode="a") as f:
        json.dump(args.__dict__, f)
        
    if args.lambda_z > 0:
        args.input_noise = True
 
    ### load data ###
    args.norm_param_file = f"{args.output_dir}/norm_params.txt"
    source, target, source_val, target_val = load_Lya_data(args.data_dir, ndata=10000, n_feature_in=2, norm_param_file=args.norm_param_file, is_train=True, device=None)
    dataset = MyDataset(source, target)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
    print("# batch_size: ", args.batch_size)
    print("# source data: ", np.shape(source))
    print("# target data: ", np.shape(target))

    args.n_feature_in = source.shape[1]
    args.n_feature_out = target.shape[1]
    ndim = len(source.shape) - 2

    ### define model ###
    args.model = "pix2pix" if ndim == 2 else "vox2vox"
    print("# model: ", args.model)

    args.isTrain = True
    model = MyModel(args)
    model.setup(args, verbose=True) #set verbose=True to show the model architecture
    
    if ndim == 2:
        source_size = source[:args.batch_size].shape
        if args.input_noise:
            source_size = (source_size[0], source_size[1] + 1, source_size[2], source_size[3])
        summary(model.netG, input_size=source_size, col_names=["input_size", "output_size", "num_params"], device=device)
        summary(model.netD, input_size=(torch.cat([source,target],dim=1))[:args.batch_size].shape, col_names=["input_size", "output_size", "num_params"], device=device)
    elif ndim == 3:
        source_size = source[:args.batch_size].shape
        if args.input_noise:
            source_size = (source_size[0], source_size[1] + 1, source_size[2], source_size[3], source_size[4])
        summary(model.netG, input_size=source_size, col_names=["input_size", "output_size", "num_params"], device=device)
        summary(model.netD, input_size=(torch.cat([source,target],dim=1))[:args.batch_size].shape, col_names=["input_size", "output_size", "num_params"], device=device)

    ### save arguments ###
    fname = f"{args.output_dir}/args.json"
    with open(fname, "w") as f:
        json.dump(vars(args), f)
    print(f"# Arguments saved to {fname}")

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
                    src_ref = torch.unsqueeze(source[args.save_image_irun], 0).to(device)
                    tgt_ref = torch.unsqueeze(target[args.save_image_irun], 0).to(device)
                    model.set_input([src_ref, tgt_ref])
                    fid = "{}/iter_{:d}".format(args.output_dir, total_iters)
                    model.save_test_image(args, fid, overwrite=True) 
                else:
                    fid = "{}/iter_{:d}".format(args.output_dir, total_iters)
                    visuals = model.get_current_visuals()
                    for iout in range(args.n_feature_out):
                        fname = "{}_{:d}.fits".format(fid, iout)
                        save_image(visuals["fake_B"][0][iout], fname, 1., overwrite=True)
                print("# save {}_*.fits".format(fid))
            
            del src, tgt
            torch.cuda.empty_cache()

        print('# End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.n_epochs + args.n_epochs_decay, time.time() - epoch_start_time))

    print('# saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
    model.save_networks('latest')

if __name__ == "__main__":
    main()
