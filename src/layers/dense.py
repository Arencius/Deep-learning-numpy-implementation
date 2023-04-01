import numpy as np
from src.layers.layer import Layer


class DenseLayer(Layer):
    def __init__(self,
                 input_neurons: int,
                 output_neurons: int):
        """
        Fully connected layer. This class inherits from Layer class
        :param input_neurons: number of neurons (features) given as an input to the layer
        :param output_neurons: number of output neurons of the layer
        """
        super().__init__()

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.input_data = None

        self.weights = self._initialize_weights()
        self.biases = np.zeros(shape=(1, self.output_neurons))

    @property
    def parameters(self):
        return (self.input_neurons * self.output_neurons) + self.output_neurons

    @property
    def output_shape(self):
        return (1, self.output_neurons)

    def _initialize_weights(self):
        return np.random.normal(size=(self.input_neurons, self.output_neurons))

    def forward(self, x):
        self.input_data = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, output_gradients):
        raise NotImplementedError
