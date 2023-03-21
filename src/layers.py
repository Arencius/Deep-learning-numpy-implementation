import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, input_neurons,
                 output_neurons,
                 activation=None,
                 use_bias=True,
                 trainable=True):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.activation = activation
        self.trainable = trainable

        self.weights = np.random.normal(size=(self.input_neurons, self.output_neurons))
        self.biases = np.random.normal(size=(self.output_neurons,)) if use_bias else np.zeros((self.output_neurons, ))

    def __str__(self):
        return '''Abstract layer class.'''

    @abstractmethod
    def forward(self, x):
        return x

    @abstractmethod
    def backward(self, output_error, learning_rate):
        pass
