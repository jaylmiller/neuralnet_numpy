import pandas as pd
import numpy as np
from neural_net import NeuralNet
from linear import LinearLayer
import time
import sys


default_set = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

batchsize = 64
dataframe = pd.read_csv(default_set)
data = dataframe.to_numpy()
inputs = data[:,:-1].astype('float')
labels = pd.factorize(data[:,-1])[0]
output_size = np.max(labels)+1
net = NeuralNet(inputs.shape[1], output_size, [LinearLayer(32)])


while True:
    correct_count = 0
    for i in range(batchsize):
        idx = np.random.randint(0, inputs.shape[0])
        inputvec = inputs[idx][np.newaxis, :]
        targetvec = np.zeros(output_size)
        targetvec[labels[idx]] = 1.0
        targetvec = targetvec[:, np.newaxis]
        output = net.forward_pass(inputvec)
        net.backward_pass(targetvec)
        pred = np.argmax(output)
        if pred == labels[idx]:
            correct_count += 1
    print('epoch accuracy =', correct_count/batchsize)
    time.sleep(1)
