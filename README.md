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

- `xai_exp_analysis.ipynb` provides a collection of plots to analyze a single XAI experiment. The current one focuses on `faint_ha`, but could be replaced with another one.
- `xai_analysis_single_signal.ipynb` analyzes two XAI experiments side by side. Currently, these are `only_ha` and `only_oiii`, but could be replaced with two other experiments.
- `xai_occlusion_eval.ipynb` can only be used to evaluate the occlusion sensitivity XAI experiment.
- `compare_to_ref.ipynb` provides a quick overview of any XAI experiment or newly trained GAN by comparing it to a reference model. 
- `plotting_maps_different_GANs.ipynb` is a simple Jupyter notebook only plotting the maps for training experiments. It is useful to visually compare maps.
- `xai_power_spectrum_development.ipynb` is intended to compare the power spectrum of a generated (perturbed) image with the test dataset statistics. This notebook is unfinished.

Please set the Python path for running the Jupyter notebooks. For example like this. 
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
