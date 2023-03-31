import numpy as np
from src.layers.layer import BaseLayer


class Relu(BaseLayer):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, output_gradients):
        return np.where(output_gradients >= 0, 1, 0)


class LeakyRelu(BaseLayer):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(x, self.alpha * x)

    def backward(self, output_gradients):
        return np.where(output_gradients >= 0, 1, self.alpha)
