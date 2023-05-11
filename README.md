# conditional GAN

cGAN model based on ![this](https://github.com/eriklindernoren/PyTorch-GAN)

## Requirement

- Python 3.8+

Install the following libraries with `pip`.
- torch==1.12.0
- torchvision==0.13.0
- torchinfo
- tqdm
- astropy

## Dataset Preparation

Deafault data directories:
- ./training_data
- ./validation_data
- ./test_data

## How to Run

```
python main.py --mode train
python main.py --mode test
```
or

```
./run.sh
```
You can change the parameters in run.sh.


To check the model performance, run plot.ipynb. Output examples:
![loss](https://github.com/knmoriwaki/MergerTree-to-SFR/blob/images/loss.png) ![test](https://github.com/knmoriwaki/MergerTree-to-SFR/blob/images/test.png)

You can check the model structure in the output file ./tmp/out_{model_name}.log. Also see model.py for details. 


- Input shape: (batch size, 1, pixel size, pixel size)

- Output shape: (batch size, 1, pixel size, pixel size)


## References


## Current Issues

- If you encounter CUDNN_STATUS_EXECUTION_FAILED:
I don't know the details, but it is solved when I re-launch the console.

- Somehow epoch_count is not set to the default value in add_argument, so you should manually set it.







