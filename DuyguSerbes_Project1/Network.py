import numpy as np

class network():
    def __init__(self, vocab_size=250, embed_dim=16, hidden_dim=128, l_rate=0.001 ):
        print("starting network")
        self.embed_dim=embed_dim
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        #Weight and hidden layer initialization
        self.weights_input_to_embedding = np.random.normal(0.0, self.vocab_size**-0.5,(self.vocab_size, self.embed_dim))
        print("w1 dim:" ,self.weights_input_to_embedding.shape)
        self.weights_embedding_to_hidden = np.random.normal(0.0, self.embed_dim** -0.5,(self.embed_dim*3, self.hidden_dim))
        print("w2 dim:", self.weights_embedding_to_hidden.shape)
        self.bias_embedding_to_hidden = np.zeros(( hidden_dim)) * np.sqrt(1 / self.embed_dim)
        print("b1 dim:", self.bias_embedding_to_hidden.shape)
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_dim ** -0.5,(self.hidden_dim, self.vocab_size))
        print("w3 dim:", self.weights_hidden_to_output.shape)
        self.bias_hidden_to_output = np.zeros((vocab_size)) * np.sqrt(1 / self.hidden_dim)
        print("b2 dim:", self.bias_hidden_to_output.shape)
        self.lr = l_rate



    #Sigmoid activation function
    def sigmoid(self, output):
        return 1. / (1. + np.exp(-output))

    #Softmax activation function
    def softmax(self, output):
        return np.exp(output) / (np.sum(np.exp(output), axis=1, keepdims=True ))

    #Prepare input for embedding layer by transforming index number to one-hot vector representation
    def embedding(self, input):
        self.one_hot_matrix = np.zeros((input.shape[0], self.vocab_size))
        for i in range(input.shape[0]):
            self.one_hot_matrix[i, input[i]] = 1.0
        return self.one_hot_matrix
    #Embedding layer by multiplied one-hot vectors with W1
    def embedding_layer(self, word_index):
        self.one_hot_matrix = self.embedding(word_index)
        embed=np.dot(self.one_hot_matrix, self.weights_input_to_embedding)
        return embed


    def forward(self, input):

        self.input=input
        #Embedding layer
        for i in range(self.input.shape[1]):
            self.embed=self.embedding_layer(input[:,i])
            if i==0:
                self.embed_total=self.embed
            else:
                self.embed_total=np.concatenate((self.embed_total,self.embed), axis=1)

        #Hidden Layer
        output=np.dot(self.embed_total, self.weights_embedding_to_hidden)
        self.output1=output+self.bias_embedding_to_hidden.T
        self.hidden= self.sigmoid(self.output1)

        #Output Layer
        output = np.dot(self.hidden, self.weights_hidden_to_output)
        self.output2 = output + self.bias_hidden_to_output.T
        self.y_prob = self.softmax(self.output2)
        return self.y_prob

    #Top-3 Accuracy Calculation
    def top_3_acc(self):
        self.top3acc=0
        prob = np.array(self.prob)
        sorted_prob=prob.argsort(axis=1)

        most_prob_1 = (sorted_prob[:, -1])
        most_prob_2 = (sorted_prob[:, -2])
        most_prob_3 = (sorted_prob[:, -3])

        self.top3acc = np.sum((most_prob_1)==self.output)
        self.top3acc += np.sum((most_prob_2) == self.output)
        self.top3acc += np.sum((most_prob_3) == self.output)



    #Cross entropy loss function and accuracy
    def cross_entropy(self, prob, output):
        self.prob=prob
        self.output=output
        self.loss=0
        acc = np.sum((np.argmax(prob, axis=1) == output))
        self.top_3_acc()
        for i in range(prob.shape[0]):
            self.loss += -np.log(self.prob[i, self.output[i]])
        predicted_probability = self.prob[np.arange(len(self.prob)), self.output]
        log_preds = np.log(predicted_probability+0.0000001)
        self.loss = -1.0 * np.sum(log_preds)
        return self.loss , acc ,self.top3acc

    #Back propagation steps
    def backward(self, output, loss):
        self.output=output
        self.d_weight_input_to_embed=0
        #print(self.embedding(self.output).shape)
        self.output_one_hot=self.embedding(self.output)
        #error_rate3=(1 / (self.output.shape[0])) *(self.y_prob-self.output_one_hot)
        error_rate3 = (self.y_prob - self.output_one_hot)
        #print("error1", error_rate3.shape)
        #print(self.hidden.shape)
        self.d_weights_hidden_to_output=(np.dot(self.hidden.T,error_rate3))
        #print("dw3:",self.d_weights_hidden_to_output.shape )
        self.d_bias_hidden_to_output=np.sum(error_rate3, axis=0)
        #print("db2", self.d_bias_hidden_to_output.shape)
        self.error_rate2=np.dot(error_rate3,self.weights_hidden_to_output.T)*(self.sigmoid(self.output1)*(1-self.sigmoid(self.output1)))
        #print("e2", self.error_rate2.shape)
        self.d_weights_embed_to_hidden=np.dot(self.embed_total.T,self.error_rate2)
        #print("dw2", self.d_weights_embed_to_hidden.shape)
        self.d_bias_embed_to_hidden =  np.sum(self.error_rate2, axis=0)
        #print("db1", self.d_bias_embed_to_hidden.shape)
        self.error_rate1=np.dot(self.error_rate2, self.weights_embedding_to_hidden.T)
        #print("e3", self.error_rate1.shape)
        #print(self.input.shape)

        for i in range(self.input.shape[1]):
            #print(self.embedding(self.input[:,i]).shape)
            self.d_weight_input_to_embed += np.dot(self.embedding(self.input[:,i]).T,
                                         self.error_rate1[:, i * self.embed_dim:(i + 1) * self.embed_dim])
            #print(self.d_weight_input_to_embed.shape)
        self.update_weights()

    #Weight Update
    def update_weights(self):
        #print("weight update")

        self.weights_hidden_to_output-=self.lr*self.d_weights_hidden_to_output
        self.bias_hidden_to_output-=self.lr*self.d_bias_hidden_to_output
        self.weights_embedding_to_hidden-=self.lr*self.d_weights_embed_to_hidden
        self.bias_embedding_to_hidden-=self.lr*self.d_bias_embed_to_hidden
        self.weights_input_to_embedding-=self.lr*self.d_weight_input_to_embed







































