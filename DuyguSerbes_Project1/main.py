import numpy as np
from Network import network
import pickle
import eval
import matplotlib.pyplot as plt
import tsne

def plot_graph(x1, y1, x2, y2 , name, x_axis , y_axis,):

    #plot accuracy, loss values
    plt.plot(x1,y1, label="training")
    plt.plot(x2,y2, label="validation")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(name)
    plt.legend()
    plt.show()
    




def training_and_validation(train_input, train_output, valid_input, valid_output):
    #Determine hyperparameters
    minibatch_size = 64
    epoch = 100
    max_loss=100000000
    vocab_size = 250
    embed_dim = 16
    hidden_dim = 128
    learning_rate = 0.01

    #initialize network
    nn = network(vocab_size, embed_dim, hidden_dim, learning_rate)

    #initialize variables
    training_acc_list=[]
    valid_acc_list=[]
    training_loss_list=[]
    valid_loss_list=[]
    training_acc_list_batch=[]
    valid_acc_list_batch=[]
    training_loss_list_batch=[]
    valid_loss_list_batch=[]
    mini_batch_list_train=[]
    mini_batch_list_valid=[]
    training_acc_list_top3 = []
    validation_acc_list_top3 = []

    epoch_list = list(range(1, epoch+1))
    num1=1
    num2=1

    for e in range(epoch):

        training_total_acc = 0
        training_total_loss = 0
        valid_total_acc = 0
        valid_total_loss = 0
        training_top3_acc=0
        valid_top_3_acc=0

        print("Current Epoch: ", e+1)

        #Shuffle data
        np.random.seed(42)
        perm = np.random.permutation(train_input.shape[0])
        train_input = train_input[perm, :]
        train_output = train_output[perm]



        #Training
        #Start minibatch loop for training
        for i in range(train_input.shape[0] // minibatch_size):

            #Determine minibatches
            input_mini = train_input[i * minibatch_size:(i + 1) * minibatch_size]
            output_mini = train_output[i * minibatch_size:(i + 1) * minibatch_size]

            #Feedforward and get softmax output
            y_pred = nn.forward(input_mini)

            #Loss function
            loss, acc, top3= nn.cross_entropy(y_pred, output_mini)

            #Backpropagation
            nn.backward(output_mini, loss)

            #Update loss and accuracy
            training_total_loss += loss
            training_total_acc += acc
            training_top3_acc += top3
            training_acc_list_batch.append(acc/minibatch_size)
            training_loss_list_batch.append(loss/minibatch_size)
            mini_batch_list_train.append(num1)
            num1+=1

        training_acc_list.append(training_total_acc/ train_input.shape[0])
        training_loss_list.append(training_total_loss / train_input.shape[0])
        training_acc_list_top3.append(training_top3_acc/train_input.shape[0])
        #Print epoch result
        print("Training Accuracy: ", round(training_total_acc / train_input.shape[0], 4), " Training Loss: ",
              round(training_total_loss / train_input.shape[0], 4), " Training Top-3 Accuracy: ", round(training_top3_acc / train_input.shape[0], 4))

        # Valdation
        # Start minibatch loop for validation no backpropagation
        for i in range(valid_input.shape[0] // minibatch_size):
            #Here shuffling meaningless, determine minibatches
            input_mini = valid_input[i * minibatch_size:(i + 1) * minibatch_size]
            output_mini = valid_output[i * minibatch_size:(i + 1) * minibatch_size]

            # Feedforward and get softmax output
            y_pred = nn.forward(input_mini)

            # Loss function
            loss, acc, top3 = nn.cross_entropy(y_pred, output_mini)

            # Update loss and accuracy
            valid_total_acc += acc
            valid_total_loss += loss
            valid_top_3_acc+= top3
            valid_acc_list_batch.append(acc / minibatch_size)
            valid_loss_list_batch.append(loss / minibatch_size)
            mini_batch_list_valid.append(num2)
            num2 += 1

        valid_acc_list.append(valid_total_acc / valid_input.shape[0])
        valid_loss_list.append(valid_total_loss / valid_input.shape[0])
        validation_acc_list_top3.append(valid_top_3_acc/valid_input.shape[0])

        #Print epoch validation results
        print("Validation Accuracy: ", round(valid_total_acc / valid_input.shape[0], 4), " Validation Loss: ",
              round(valid_total_loss / valid_input.shape[0], 4), " Validation Top-3 Accuracy: ", round(valid_top_3_acc / valid_input.shape[0], 4))

        #When validation loss is improved save model
        if valid_total_loss < max_loss:
            max_loss = valid_total_loss
            model_dir = "./models/model.pk"
            save_model_file = open(model_dir, "wb")
            pickle.dump(nn, save_model_file)
            save_model_file.close()
            print("Current model is saving...")

    #Plot accuracy and loss values
    plot_graph(epoch_list, training_acc_list, epoch_list, valid_acc_list, "accuracy of model", "epoch", "accuracy(%)")
    plot_graph(epoch_list, training_loss_list, epoch_list, valid_loss_list, "loss of model", "epoch", "loss")
    plot_graph(epoch_list, training_acc_list_top3, epoch_list, validation_acc_list_top3, "top-3 accuracy of model", "epoch", "accuracy")
    #plot_graph(mini_batch_list_train, training_acc_list_batch, mini_batch_list_valid, valid_acc_list_batch, "accuracy of model (minibatches)", "minibatch steps", "accuracy")
    #plot_graph(mini_batch_list_train, training_loss_list_batch,  mini_batch_list_valid,valid_loss_list_batch, "loss of model (minibatches)", "minibatch steps", "loss")


    return model_dir



def main():


    # Load train data
    train_input = np.load("./data/train_inputs.npy")
    train_output = np.load("./data/train_targets.npy")

    # Load validation data
    valid_input = np.load("./data/valid_inputs.npy")
    valid_output = np.load("./data/valid_targets.npy")

    # Load test data
    test_input = np.load("./data/test_inputs.npy")
    test_output = np.load("./data/test_targets.npy")

    #Load vocabulary
    vocabs = np.load("./data/vocab.npy")

    print("Number of training input:", train_input.shape[0])
    print("Number of valid input:", valid_input.shape[0])
    print("Number of training input:", test_input.shape[0])
    print("Number of word:", len(vocabs))

    model_directory="./model.pk"

    #Run training loop and save most successful model
    model_directory=training_and_validation(train_input, train_output, valid_input, valid_output)
    print(model_directory)

    #Run evaluation code
    eval.evaluation( test_input, test_output, model_directory)  #It can be run within itself, no need to run in main.py

    #Run TSNE code
    tsne.tsne(model_directory, vocabs)


    #Find the most probable words
    eval.find_most_probable(["city", "of" ,"new"], vocabs, model_directory)
    eval.find_most_probable(["life", "in", "the"] ,vocabs, model_directory)
    eval.find_most_probable(["he", "is", "the"],vocabs, model_directory)




main()
