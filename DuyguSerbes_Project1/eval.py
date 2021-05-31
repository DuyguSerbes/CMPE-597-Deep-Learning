import pickle
import numpy as np
import math
#from sklearn.metrics import confusion_matrix, plot_confusion_matrix
#import matplotlib.pyplot as plt
#import seaborn as sns

def load(name):
    f = open(name, 'rb')
    model = pickle.load(f)
    f.close()
    return model


#Please write test input and target values path
test_input_p="./data/test_inputs.npy"
test_output_p="./data/test_targets.npy"

#Load vocabulary
vocab = np.load("./data/vocab.npy")

#Load data
test_input_default = np.load(test_input_p)
test_output_default = np.load(test_output_p)

def find_most_probable(words,vocabs=vocab, model_directory="./models/model.pk"):
    #Find index of given words
    sentence = [np.where(vocabs == words[0])[0][0], np.where(vocabs == words[1])[0][0], np.where(vocabs == words[2])[0][0]]
    sentence = np.array(sentence).reshape((1, 3))
    model = load(model_directory)
    prob = model.forward(sentence)
    prob = np.array(prob)
    #Find most probable word index
    most_prob_1 = (prob.argsort()[0, -1])
    most_prob_2 = (prob.argsort()[0, -2])
    most_prob_3 = (prob.argsort()[0, -3])
    most_prob_4 = (prob.argsort()[0, -4])
    most_prob_5 = (prob.argsort()[0, -5])
    print("for sentence," , words[0], words[1], words[2],", most probable next word listed below in descending order:")
    #list the vocabs
    print(vocabs[most_prob_1])
    print(vocabs[most_prob_2])
    print(vocabs[most_prob_3])
    print(vocabs[most_prob_4])
    print(vocabs[most_prob_5])





def evaluation(test_input=test_input_default, test_output=test_output_default, model_directory="./models/model.pk"):

    test_batch = 32  # for fast calculation
    batches = math.ceil(test_input.shape[0] / test_batch)
    test_accuracy = 0
    test_loss = 0
    test_top3=0

    # Load model file
    model = load(model_directory)
    pred=np.array

    for i in range(batches):
        input_mini=test_input[i*test_batch:min((i+1)*test_batch, test_input.shape[0])]
        output_mini = test_output[i * test_batch:min((i + 1) * test_batch, test_input.shape[0])]
        y_pred=model.forward(input_mini)
        a=(np.argmax(y_pred, axis=1))
        if i==0:
            pred=a
        else:
            pred=np.concatenate((pred,a), axis=None)
        loss, acc, top3 = model.cross_entropy(y_pred, output_mini)
        test_accuracy += acc
        test_loss+=loss
        test_top3+=top3
    #Result of accuracy, top3 accuracy and loss
    test_accuracy=test_accuracy/test_input.shape[0]
    test_top3=test_top3/test_input.shape[0]
    test_loss=test_loss/test_input.shape[0]
    print("Test Accuracy:", test_accuracy , "Test Loss:", test_loss , "Test Top-3 Accuracy:", test_top3)

    #matrix=confusion_matrix(pred,test_output_default)
    #print(matrix)
    #print(confusion_matrix)

 #   fig, ax = plt.subplots(figsize=(11, 9) )
 #   sns.heatmap(matrix,annot=False ,vmin=0, vmax=100)
 #   plt.show()



if __name__ == "__main__":
    evaluation()
    model_directory="./models/model.pk"
    find_most_probable(["city", "of" ,"new"])
    find_most_probable(["life", "in", "the"])
    find_most_probable(["he", "is", "the"])

