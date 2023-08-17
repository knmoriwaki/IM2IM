# conditional GAN

cGAN model based on pix2pix [(Isola et al. 2016)](https://github.com/eriklindernoren/PyTorch-GAN)

## Requirement

- Python 3.8+

Install the following libraries with `pip`.
- torch==1.12.0
- torchvision==0.13.0
- torchinfo
- tqdm
- astropy

## How to Run

Run training and testing can be run using 
```
./run.sh
```
The implementation in this branch allows training ~~`gan_mode=vanilla`~~, ~~`gan_mode=wgan`~~, and `gan_mode=wgangp`. 

Use `plot.ipynb` to check the model performance. More scripts for checking the model performance can be found in branch #4. 
