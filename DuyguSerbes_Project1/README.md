# CMPE-597-Deep-Learning

A neural language model using a multi-layer perceptron from scratch. The model takes sequence of 3 words and predicts the forth word.

## Overview

The neural network consists of three layer. It predicts the next word according to given 3 words. First layer is 16-dimensional embedding layer, second layer is 128-dimensional hidden layer with sigmoid activation function whereas third layer is 250-dimensional output layer with softmax activation function. Cross-entropy loss function is used as a criterion to calculate loss of network.

You are able to evaluate pre-trained network using `eval.py/` using `data/test_inputs.npy/` as input matrix and `data/test_targets.npy/` as output matrix. 

You are able to run `tsne.py/` to see the 2-D plot of embeddings using pre-trained network parameters. 

All functions used in `Network.py/` explained in the document can be found in `CMPE597_project_1_duygu_serbes_2020700171.pdf/` directory. All training, validation, test performances reported with plots. 

*Please note that though this network may work on various platforms, it has only been tested on an Ubuntu 18.04 system.*

## Installation Dependencies:
* Python 3.X
* numpy
* sckit-learn
* matplotlib
* pandas
* sklearn

## How to Run?

Create conda envrionment:
```
cd DuyguSerbes_Project1
conda create --name project_1_env
conda activate project_1_env
pip install -r requirements.txt
```

Make sure all data files are located in the `data/` folder. Because of restricted size of moodle, `data` folder sent by instructor cannot be added `DuyguSerbes_Project1` file. In order to move the `data/` folder, below command can be used:

```
sudo mv <current_path_of_data_folder> DuyguSerbes_Project1

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
* `data/` - all data sources required for training/validation/testing. `test_inputs.npy`, `test_outputs.npy`, `train_inputs.npy`,`train_outputs.npy`, `valid_inputs.npy`,`valid_outputs.npy` and `vocab.npy` should be put in that directory.
* `report/` - report of project in pdf format and all required plots. 
* `main.py/` - python code to load the dataset, shuffle the training data and divide it into mini-batches, the loop for the epoch and iterations, and evaluate the model on validation set during training
* `Network.py/`-python code for the forward, backward propagation, and the activation functions.
* `eval.py/` - python code to load the learned network parameters and evaluate the model on test data. Also, you can run the code to find the most probable next five word of new sequence, the words are given in descending order. 
* `tsne.py/` - python code to create 2-D plot of the embeddings using t-SNE. 
* `models/` - pretrained models directory. `models/model.pk` is located here can be used for evaluation and tsne functions.



