from autoencoder import AutoEncoder
from linear import LinearLayer
import numpy as np
from utils import sigmoid, column_vector, cross_entropy, sum_of_squares_error
import random


class NeuralNet:
    """Deep neural network.
    """
    def __init__(self,input_size, output_size, hidden_layers=[], output_activ_func=sigmoid,
        loss_func=sum_of_squares_error, l_rate=0.05):
        """
        args:
            input_size - size of input
            hidden_layers - list of hidden layers (objects),
                should be initialzed, but not pretrained
            output_activ_func - activation function for output
            output_size - number of output nodes
            loss_func - the loss function to train the net
            l_rate - learning rate of whole network (hidden layer
                learning rates during pretraining can be different)
        """

        self.l_rate = l_rate
        self.input_size = input_size
        self.input = np.matrix(np.zeros(input_size)).T
        self.output = np.matrix(np.zeros(output_size)).T
        self.output_activ_func = output_activ_func
        self.hidden_layers = hidden_layers
        self.loss_func = loss_func
        self.output_size = output_size
        self.errs = []

        # list of weight matrices for hidden layers
        # nth weight matrix is from the nth hidden layer to the n+1th
        self.hidden_weights = []
        for i, layer in enumerate(self.hidden_layers):
            if i == 0:
                next_size = layer.size
                prev_size = self.input_size
            else:
                next_size = layer.size
                prev_size = self.hidden_layers[i-1].size

            W_new = np.random.normal(loc=0, scale=1.0/np.sqrt(next_size), 
                size=(next_size, prev_size))
            self.hidden_weights.append(W_new)
        last_hid_size = self.hidden_layers[-1].size
        self.W_oh = np.random.normal(0, 1/np.sqrt(output_size), (output_size, last_hid_size))
        # bias vector for output
        self.B_o = np.random.normal(loc=0, scale=1, size=(output_size, 1))
        self.pretrain_hidden([])

    def backward_pass(self, target):
        """ Perform a backward pass on the network given the
        negative of the derivative of the error with respect to the output.
        """
        self.errs.append(self.loss_func(self.output, target))
        out = column_vector(self.output)

        dE_dOut = -1*self.loss_func(out, column_vector(target), True)

        dOut_dNet = self.output_activ_func(out, True, True)
        delta_out = np.multiply(column_vector(dE_dOut), column_vector(dOut_dNet))
        deltas = []
        t = np.dot(self.W_oh.T, delta_out)
        mat = np.dot(delta_out, self.hidden_layers[-1].a_h.T)
        self.W_oh += self.l_rate*mat
        for i in range(1, len(self.hidden_layers)+1):
            layer = self.hidden_layers[-i]

            delta = np.multiply(t, layer.activ_func(layer.a_h, True, True))
            deltas.append(delta)
            if i < len(self.hidden_layers):
                t = np.dot(layer.weights.T, delta)
                # t = np.dot(self.W_oh.T, delta_out)
                mat = np.dot(delta, self.hidden_layers[-(i+1)].a_h.T)
                layer.weights += self.l_rate*mat

    def forward_pass(self, input):
        """ Perform a forward pass on the network given
        an input vector.
        """
        self.input = input
        for layer in self.hidden_layers:
            self.input = layer.encode(self.input)
        self.output = np.dot(self.W_oh, column_vector(self.input))
        self.output = self.output_activ_func(self.output+self.B_o)
        return self.output

    def pretrain_hidden(self, training_data, n_epochs=300):
        """ Pretrain each hidden layer given the training_data.
        Parameters effecting hidden layer training are within
        each hidden layers object
        """
        # pretrain each layer seperately, bottom up
        for weights, layer in zip(self.hidden_weights, self.hidden_layers):
            layer.train(training_data, weights, n_epochs)
            # training_data = layer.new_data_representation(training_data)
        print(self.W_oh.shape)
        # after pretraining, clamp learned values
        # self.hidden_weights = [l.weights for l in self.hidden_layers]
        # self.hidden_biases = [l.B_h for l in self.hidden_layers]


if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    np.set_printoptions(suppress=True)
    in_size = 32
    layers = [LinearLayer(10)]
    net = NeuralNet(in_size, 6, layers)
    size = 10
    data = [np.random.normal(loc=0, scale=2, size=(1, in_size)) for i in range(size)]
    net.pretrain_hidden(data)
    net.forward_pass(data[0])
    d = np.array([1, 1, 0, 0, 1, 1])
    d = d[:, np.newaxis]
    net.backward_pass(d)
    print(net.output)
