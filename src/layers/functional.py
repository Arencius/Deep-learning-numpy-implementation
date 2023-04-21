import numpy as np
from src.layers.layer import Layer


class FLattenLayer(Layer):
    def forward(self, x: np.array):
        return np.array([matrix.flatten() for matrix in x])
