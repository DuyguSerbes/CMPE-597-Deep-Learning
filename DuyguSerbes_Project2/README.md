# CMPE 597 Deep Learning Project 2

## Overview
In this project, CNN architecture is implemented using PyTorch deep learning library. As training, validation and test data CIFAR10 is used. The final architecture has 3 CNN layer which has 48, 96 and 192 feature maps respectively and has 2 fully connected layer at the end. After second CNN layer and first fully connected layer dropout is used with 0.1 and 0.3 dropout probability. Batch normalization is used after all layers except regression layer which is found at the end. As a result of experiments, Adam optimizer is used with decreasing learning rate at certain epoch number. 

## Implementation 
*Please note that though this network may work on various platforms, it has only been tested on an Ubuntu 18.04 system.* 

First, create a conda env for your system and activate it:
```bash
cd DuyguSerbes_Project2
conda env create -f environment.yml
conda activate project2
```
Run the all training, validation, test function in a one command (it also loads the required data):
```
python3 main.py

```
Run the eval.py with pretrained network to calculate test accuracy and loss values by giving model directory as an argument (it also loads the required test data):
```
python3 eval.py <model_directory>

```
For instance:
```
python3 eval.py ./model.pt

```

If t-SNE plots wants to be created, the line between 186 and 199 in main.py should be comment out status. 


## Structure
The directory structure is as follows.

    .
    ├── main.py                      # training, validation and test of model
    ├── model.py                     # model architecture for using training, validation and test
    ├── eval.py                      # provide evaluation on test data with pre-trained network. 
    ├── DuyguSerbes_Project2.pdf     # report of project 2
    ├── README.txt                   # the text file needed for implementation
    ├── environment.yml              # the YAML file needed for environment creation
    ├── data                         # it will be created after first run
   
