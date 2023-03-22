import numpy as np
from src.layers.layer import BaseLayer


class Relu(BaseLayer):
    def forward(self, x):
        x[x < 0] = 0
        return x

    def backward(self, output_gradients):
        return output_gradients
