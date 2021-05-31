import Network as nn
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns

#load the model
def load(name):
    f = open(name, 'rb')
    model = pickle.load(f)
    f.close()

    return model
#load the vocabs
vocabs = np.load("./data/vocab.npy")


def tsne(model_direc="./models/model.pk", vocab=vocabs):

    model = load(model_direc)
    #Get embedding vector of each word
    for index, word in enumerate(vocab):
        element=np.array([index]).reshape((1,1))
        embed_layer=model.embedding_layer(element)
        if index==0:
            matrix=embed_layer
        else:
            matrix=np.append(matrix, embed_layer, axis=0)
        #print(matrix.shape)
    #tsne results
    plt.figure(figsize=(18, 18))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(matrix)
    #print(tsne_results)
    x_ = tsne_results[:, 0]
    y_ = tsne_results[:, 1]
    # display scatter plot
    plt.scatter(x_, y_)
    #put labels on the plot
    for label, x, y in zip(vocab, x_, y_):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_.min() - 1, x_.max() + 1)
    plt.ylim(y_.min() -1 , y_.max() + 1)
    plt.show()
    plt.savefig("tsne.png")



if __name__ == "__main__":
    tsne()
