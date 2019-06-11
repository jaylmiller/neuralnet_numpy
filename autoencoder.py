""" Train a denoising-autoencoder to be used as a hidden layer in the network.
"""

import numpy as np
from utils import *
import random


class AutoEncoder:

    def __init__(self, size, activ_func=sigmoid, l_rate=.02, corrupt_rate=.1):
        """Initializer.

        Args:
            size - number of units
            activ_func - activation function on hidden units
            l_rate - learning rate (for pretraining)
        """
        self.size = size
        self.activ_func = sigmoid
        self.loss_func = sum_of_squares_error
        self.l_rate = l_rate
        self.corrupt_rate = corrupt_rate

    def train(self, input_data, initial_weights, epochs=300):
        """Train the autoencoder.

        Args:
            input_data - Data to use
            initial_weights - initial weight matrix, passed from the DeepNN
            epochs - epochs to run
            learnrate - learning-rate
        Returns:
            tuple: new weights and bias
        """
        self.weights = initial_weights
        self.B_h = np.random.normal(loc=0, scale=.5,
                                    size=(self.weights.shape[0], 1))
        self.B_o = np.random.normal(loc=0, scale=.5,
                                    size=(self.weights.shape[1], 1))

        for i in range(epochs):
            random.shuffle(input_data)
            err_per_epoch = 0
            for data_vector in input_data:
                if self.corrupt_rate > 0:
                    data_vector = self.noise(data_vector)
                # data_vector = data_vector.T  # want column matrix
                hidden = self.encode(data_vector)  # get encoding
                out = self.decode(hidden)  # get decoding
                error = self.loss_func(column_vector(out), column_vector(data_vector))
                err_per_epoch += error
                # backprop time
                dE_dOut = -1*self.loss_func(column_vector(out), column_vector(data_vector), True)
                dOut_dNet = self.activ_func(column_vector(out), True, True)
                # print dE_dOut.shape
                delta_out = np.multiply(dE_dOut, dOut_dNet)
                # print delta_out.shape
                t = np.dot(self.weights, delta_out)
                delta_hidden = np.multiply(t, self.activ_func(hidden,
                                                              True, True))
                delta_Woh = self.l_rate * np.dot(delta_out, hidden.T)
                delta_Bo = self.l_rate * delta_out
                delta_Bh = self.l_rate * delta_hidden

                # update weights and bias
                self.weights = self.weights + delta_Woh.T
                self.B_o = self.B_o + delta_Bo
                self.B_h = self.B_h + delta_Bh
            # print err_per_epoch/100
        self.bias = self.B_h
        return (self.weights, self.bias)

    def test(self, data):
        total_err = 0
        for data_vector in data:
            x = self.decode(self.encode(data_vector.T))
            total_err += self.loss_func(x, data_vector.T)
        print(total_err/len(data))

    def new_data_representation(self, data):
        """Return a transformed version of the inputted data
        where the transformation is the encoding learned by this layer.
        """
        return [self.encode(v) for v in data]


    def encode(self, data_vector):
        """Encode a single data_vector
        """
        # make sure its a column vector
        data_vector = column_vector(data_vector)
        self.net_h = np.dot(self.weights, data_vector)+self.B_h
        self.a_h = self.activ_func(self.net_h)
        return self.a_h

    def decode(self, encoded_vector):
        """Decode an encoded_vector
        """
        encoded_vector = column_vector(encoded_vector)
        self.net_o = np.dot(self.weights.T, encoded_vector)+self.B_o
        return self.activ_func(self.net_o)

    def noise(self, x):
        p = np.random.rand(*x.shape)
        mask = np.float32(p >= self.corrupt_rate)
        return np.multiply(x, mask)

"""Just for testing...."""
if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    np.set_printoptions(suppress=True)
    a = AutoEncoder(10, l_rate=.2, corrupt_rate=.4)
    weights = np.random.normal(loc=0, scale=1.0/np.sqrt(10), size=(10, 20))
    size = 100
    data = [np.random.normal(loc=0, scale=2, size=(1, 20)) for i in range(size)]
    size = 5
    test = [np.random.normal(loc=0, scale=2, size=(1, 20)) for i in range(size)]
    # data = [np.ones((1, 20)) for i in range(5)]
    # data.extend([np.zeros((1, 20)) for i in range(5, 10)])
    a.train(data, weights, 500)
    print(a.new_data_representation(data)[0].shape)