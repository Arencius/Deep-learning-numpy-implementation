import numpy as np
from src.layers.layer import Layer


class FLattenLayer(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape

    @property
    def output_shape(self) -> tuple:
        return np.prod(self.input_shape),

    def forward(self, x: np.array):
        self.input_shape = x.shape
        return np.reshape(x, (self.input_shape[0], -1))

    def backward(self, output_gradients):
        return np.reshape(output_gradients, self.input_shape)


class DropoutLayer(Layer):
    def __init__(self, p=0.5, trainable=True):
        super().__init__()
        self.p = p
        self.trainable = trainable
        self._mask = None

    def forward(self, x):
        if self.trainable:
            self._mask = np.random.binomial(1, self.p, size=x.shape)
            x *= self._mask

        return x

    def backward(self, output_gradients):
        return output_gradients * self._mask
