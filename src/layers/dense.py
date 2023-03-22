import numpy as np
from src.layers.layer import BaseLayer


class DenseLayer(BaseLayer):
    def __init__(self,
                 input_neurons,
                 output_neurons):
        super().__init__()

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.weights = np.random.normal(size=(self.input_neurons, self.output_neurons)) * 0.1
        self.biases = np.zeros(shape=(1, self.output_neurons))
        self.parameters = (self.input_neurons * self.output_neurons) + self.output_neurons

    def forward(self, x):
        self.layer_output = np.dot(x, self.weights) + self.biases
        return self.layer_output

    def backward(self, output_gradients):
        raise NotImplementedError
