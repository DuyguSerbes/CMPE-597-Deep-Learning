# CMPE-597-Deep-Learning

A neural language model using a multi-layer perceptron from scratch. The model takes sequence of 3 words and predicts the forth word.

## Overview

The neural network consists of three layer. It predicts the next word according to given 3 words. First layer is 16-dimensional embedding layer, second layer is 128-dimensional hidden layer with sigmoid activation function whereas third layer is 250-dimensional output layer with softmax activation function. 

*Please note that though this network may work on various platforms, it has only been tested on an Ubuntu 18.04 system.*

## Installation Dependencies:
* Python 3.X
* numpy
* pickle
* matplotlib
* math


## How to Run?

Create conda envrionment:
```
conda create --name my_projetct_env
conda activate my_project_env
pip install -r requirements.txt
```

Run the all training, validation, test and tsne function in a one command:

```
python3 main.py

```

Run the eval.py with pretrained network to calculate test accuracy and loss values:
```
python3 eval.py

```

Run the tsne.py with embedding layer of pretrained network to visualize clusters of given vocabulary:
```
python3 tsne.py

```

## Structure
* `data/` - all data sources required for training/validation/testing.
* `report/` - report of project in pdf format and all required plots. 
* `main.py/` - python code to load the dataset, shuffle the training data and divide it into mini-batches, the loop for the epoch and iterations, and evaluate the model on validation set during training
* `Network.py/`-python code for the forward, backward propagation, and the activation functions.
* `eval.py/` - python code to load the learned network parameters and evaluate the model on test data.
* `tsne.py/` - python code to create 2-D plot of the embeddings using t-SNE. 

