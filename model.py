import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
 

from base_model import BaseModel
from collections import OrderedDict
from my_retanh import ReTanh

from utils import *

def MyModel(args):
    if "pix2pix" in args.model:
        model = Pix2Pix(opt=args)
    else:
        print("Error: unknown model", file=sys.stderr)
        sys.exit(1)
    return model

###############################################################################
# Helper Functions
###############################################################################

def init_net(net, gpu_ids=[]):

    if len(gpu_ids) > 0:
        if gpu_ids[0] != -1:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)

    ### Initialize network weights ### 
    def init_func(m, init_gain=0.02):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    
    return net

def define_G(opt, gpu_ids=[]):

    G = UnetGenerator(
            input_nc = opt.input_nc, 
            output_nc = opt.output_nc, 
            ngf = opt.hidden_dim_G, 
            dropout = opt.dropout,
            n_layers = opt.nlayer_G
        )
    return init_net(G, gpu_ids)

def define_D(opt, gpu_ids=[]):

    D = NLayerDiscriminator(
            input_nc = opt.input_nc + opt.output_nc,
            ndf = opt.hidden_dim_D,
            n_layers = opt.nlayer_D
            )

    return init_net(D, gpu_ids)

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):

    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):

    def __init__(self, gan_mode='vanilla'):
        super().__init__()

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'wgan']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            loss = self.loss(prediction, target_tensor.expand_as(prediction))
        elif self.gan_mode == 'wgangp' or self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, dropout, n_layers=5):
        super().__init__()

        nf_mult = min(2 ** (n_layers-1), 8)
        nf_mult_prev = min(2 ** (n_layers-2), 8)
        unet_block = UnetSkipConnectionBlock(ngf * nf_mult_prev, ngf * nf_mult, input_nc=None, innermost=True)
        for n in range(1, n_layers-1):
            nf_mult = nf_mult_prev
            nf_mult_prev = min(2 ** (n_layers - n - 2), 8)
            unet_block = UnetSkipConnectionBlock(ngf * nf_mult_prev, ngf * nf_mult, input_nc=None, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, input):
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """
    Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()

        #padw = "same" #padding="same" is not supported for strided convolutions
        kw = 4
        padw = 1 
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kw,
                             stride=2, padding=padw, bias=True)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kw, stride=2,
                                        padding=padw)
            down = [downconv]
            up = [uprelu, upconv, ReTanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kw, stride=2,
                                        padding=padw, bias=True)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kw, stride=2,
                                        padding=padw, bias=True)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=5):
        super().__init__()

        #padw = "same" is not supported for strided convolutions
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map 
        # this last layer is different from the original pix2pix, where they use flatten + linear
        # note Sigmoid function for vanilla GAN is included in the loss fuction BCEWithLogitsLoss.

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        #return self.model(x)
        return torch.mean(self.model(x), dim=(1,2,3))


class Pix2Pix(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.gan_mode = opt.gan_mode
       
        self.loss_names = [ "G_GAN", "G_L1", "D_real", "D_fake" ]
        self.visual_names = [ "real_A", "fake_B", "real_B"]

        if self.isTrain:
            self.model_names = [ "G", "D" ]
        else:
            self.model_names = [ "G" ]

        self.netG = define_G(opt, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.netD = define_D(opt, gpu_ids=self.gpu_ids)

            self.criterionGAN = GANLoss(self.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss(reduction="mean")

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*2, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = True
        self.real_A = input[0 if AtoB else 'B'].to(self.device)
        self.real_B = input[1 if AtoB else 'A'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A) 
 
    def backward_D(self):

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        if self.gan_mode == "wgan":
            for p in self.netD.parameters():
                p.data.clamp_(-0.01,0.01)

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def save_test_image(self, opt, fid, overwrite=False):
        with torch.no_grad():
            self.forward()
            for iout in range(opt.output_nc):
                fname = "{}_{:d}.fits".format(fid, iout)
                save_image(self.fake_B[0][iout], fname, opt.norm, overwrite=overwrite)
      
    def save_source_image(self, opt, fid, overwrite=False):
        with torch.no_grad():
            self.forward()
            fname = "{}_source.fits".format(fid)
            save_image(self.real_A[0][0], fname, opt.norm, overwrite=overwrite)
            for iout in range(opt.output_nc):
                fname = "{}_target_{:d}.fits".format(fid, iout)
                save_image(self.real_B[0][iout], fname, opt.norm, overwrite=overwrite)