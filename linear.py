import numpy as np
from utils import sigmoid, column_vector
import random


class LinearLayer:

    def __init__(self, size, activ_func=sigmoid):
        self.size = size
        self.activ_func = sigmoid
        self.weights = None


    def train(self, input_data, initial_weights, epochs=300):
        self.weights = initial_weights


    def encode(self, data_vector):
        """Encode a single data_vector
        """
        # make sure its a column vector
        data_vector = column_vector(data_vector)
        self.net_h = np.dot(self.weights, data_vector)
        self.a_h = self.activ_func(self.net_h)
        return self.a_h
