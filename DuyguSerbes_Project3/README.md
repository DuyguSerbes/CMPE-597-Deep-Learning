# CMPE 597 Deep Learning Project 2

## Overview
In this project, a VAE is implemented, where the encoder is an LSTM network and the decoder is a convolutional network. MNIST dataset is used, which consists of 28×28 hand written digits images. In decoder single layer LSTM is used with hidden dimension 128 whereas encoder part consists of 4 transpose convolution layer. 2D latent space model can be found in `./model.pt`. Results of experiments are listed in ther report in terms of hidden dimensionality, hyperparameters of transpose convolution layers, and latent space dimensionality experiments. Adam optimizer is used with 0.001 learning rate. Batch size is selected as 64. 

## Implementation 
*Please note that though this network may work on various platforms, it has only been tested on an Ubuntu 18.04 system.* 

First, create a conda env for your system and activate it:
```bash
cd DuyguSerbes_Project3
conda env create -f environment.yml
conda activate project3
```
Run the all training and validation. `main.py` is responsible for creation of `./output` directory and saving of recontructed validation images for each epoch. At the end of the training plot of loss values is obtained in the `./output` directory. `model.pt` is saved to current directory.  
```
python3 main.py

```
Run ` generator.py` with pretrained network to create randaomly sampled  digits. Default model directory is `./model.pt`. If you want to use another model, it should be given as command line argument. After the run the generator produces two png files in the main directory. `./generated_random_images.jpg` is the total view of 100 randomly generated images. The seperate generated images can be found in the `generated_images` folder, which is created after the first run of the `generator.py`. Also generator creates grid of sampled digits in `./generated_grid_images.jpg` directory. Furthermore, `generator.py` saves randomly produces images as 28 by 28 image in the `./generated_images` folder. 
```
python3 generator.py <model_directory>

```
Example running of generator:
```
python3 generator.py ./model.pt

```

## Structure
The directory structure is as follows.

    .
    ├── main.py                      # training and validation 
    ├── model.py                     # model architecture for encoder and decoder. Calculation of loss.
    ├── generator.py                 # generation of 100 ramdomply sampled digits and a grid of sampled digits to be saved them current directory.
    ├── model.pt                     # pretrained 2D lateral VAE model. 
    ├── DuyguSerbes_Project3.pdf     # report of project 3.
    ├── README.txt                   # the text file needed for implementation.
    ├── environment.yml              # the YAML file needed for environment creation.
    ├── generated_images             # folder generated after the first run of the `generator.py` and saves the randomly generated 28 by 28 digits seperately.
    ├── outputs                      # folder is created after the first `main.py` run, which restore reconstructed images of last mini-batch of validation
    out.csv and loss plots.
   
