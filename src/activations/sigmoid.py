import numpy as np
from src.layers.layer import BaseLayer


class Sigmoid(BaseLayer):
    def forward(self, x):
        self.layer_input = x
        return np.exp(x) / (np.exp(x) + 1)

    def backward(self, output_gradients):
        sigmoid = self.forward(self.layer_input)
        s = sigmoid * (1.0 - sigmoid)
        return output_gradients * s

class Softmax(BaseLayer):
    def forward(self, x):
        self.layer_input = x
        return np.exp(x) / np.exp(x).sum()

    def backward(self, output_gradients):
        softmax = self.forward(self.layer_input)
        s = softmax * (1.0 - softmax)
        return output_gradients * s

class Tanh(BaseLayer):
    def forward(self, x):
        self.layer_input = x
        return np.sinh(x) / np.cosh(x)

    def backward(self, output_gradients):
        tanh = self.forward(output_gradients)
        return 1.0 - (tanh ** 2)


class SiLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return x * self.sigmoid.forward(x)

    def backward(self, output_gradients):
        sigmoid = self.sigmoid.forward(output_gradients)
        return sigmoid * (1 + output_gradients * (1 - sigmoid))
