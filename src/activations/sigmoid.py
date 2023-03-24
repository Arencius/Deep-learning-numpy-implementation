import numpy as np
from src.layers.layer import BaseLayer


class Sigmoid(BaseLayer):
    def forward(self, x):
        return np.exp(x) / (np.exp(x) + 1)

    def backward(self, output_gradients):
        return NotImplementedError


class Softmax(BaseLayer):
    def forward(self, x):
        return np.exp(x) / np.exp(x).sum()

    def backward(self, output_gradients):
        raise NotImplementedError


class Tanh(BaseLayer):
    def forward(self, x):
        return np.sinh(x) / np.cosh(x)

    def backward(self, output_gradients):
        raise NotImplementedError


class SiLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return x * self.sigmoid.forward(x)

    def backward(self, *args):
        raise NotImplementedError
