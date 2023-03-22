import numpy as np
from src.layers.layer import BaseLayer


class Relu(BaseLayer):
    def forward(self, x):
        x[x < 0] = 0
        return x

    def backward(self, output_gradients):
        return output_gradients


class LeakyRelu(BaseLayer):
    def __init__(self, alpha=0.2):
        super().__init__(alpha)
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(x, self.alpha * x)

    def backward(self, output_gradients):
        return output_gradients
