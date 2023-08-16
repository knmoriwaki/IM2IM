# conditional GAN

cGAN model based on pix2pix [(Isola et al. 2016)](https://github.com/eriklindernoren/PyTorch-GAN)

## Requirements

For the training and inference 
- Python 3.8+
is needed. The environment for the inference could be called `inference_venv`. Install the following libraries with `pip`.
- torch==1.12.0
- torchvision==0.13.0
- torchinfo
- tqdm
- astropy

For the evaluation and plotting of the explainable AI experiments, a different environment is needed. 
The environment for the evaluation could be called `xai_venv`.
Install the following libraries with `pip`:
- astropy
- matplotlib          3.7.1
- numpy               1.24.3
- pandas              2.0.2

If any library is missing you can simply install it. Please also note that you might need to set the python path, especially when using Jupyter notebook. 
```
export PYTHONPATH=$HOME/IM2IM:$PYTHONPATH
```
How to set up the Jupyter notebook properly can be found in the UTAP Wiki. 

## How to Run

For training first, activate the corresponding Python environment for inference and then run the training, testing and the XAI experiments using 
```
ssh cat4
cd [IM2IM]
./srun.sh
```
You can change the parameters in run.sh. This branch only supports `gan_mode=vanilla`. 

## How to evaluate

For the evaluation a lot of different Jupyter notebooks are available.

Please set the python path for running the Jupyter notebooks. For example like this. 
```
ssh cat4
cd [IM2IM]
. [xai_venv]/bin/activate
export PYTHONPATH=$HOME/IM2IM:$PYTHONPATH
jupyter-notebook --no-browser --port=8889
```
How to set up the Jupyter notebook properly can be found in the UTAP Wiki. 

## References


## Known Issues

- Negative values for the generated images in some of the XAI experiments.
