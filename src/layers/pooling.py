import numpy as np
from src.layers.layer import Layer
from abc import ABC, abstractmethod


class PoolingLayer(Layer, ABC):
    def __init__(self,
                 input_size: tuple,
                 pool_size: int,
                 stride: int = 2):
        super().__init__()

        self.height, self.width, self.channels = input_size
        self.pool_size = pool_size
        self.stride = stride

    @property
    def output_shape(self):
        output_height = (self.height - self.pool_size) / self.stride + 1
        output_width = (self.width - self.pool_size) / self.stride + 1

        return int(output_height), int(output_width), self.channels

    @abstractmethod
    def _operation(self, image):
        raise NotImplementedError

    def _get_submatrices(self, images):
        if len(images.shape) < 4:    # adds batch dimension for single image input (h,w,c -> batch,h,w,c)
            images = images[np.newaxis, ...]

        batch_size = images.shape[0]

        shape = (batch_size, self.output_shape[0], self.output_shape[1], self.pool_size, self.pool_size, self.channels)
        strides = (images.strides[0], images.strides[1] * self.stride, images.strides[2] * self.stride,
                   images.strides[1], images.strides[2],  images.strides[3])

        sub_matrices = np.lib.stride_tricks.as_strided(images,
                                                       shape=shape,
                                                       strides=strides)
        return sub_matrices

    def forward(self, x):
        submatrices = self._get_submatrices(x)
        pooled_img = self._operation(submatrices)

        return np.reshape(pooled_img, (-1, *self.output_shape))


class MaxPoolingLayer(PoolingLayer):

    def _operation(self, image):
        return np.max(image, axis=(3, 4))


class AveragePoolingLayer(PoolingLayer):
    def _operation(self, image):
        return np.mean(image, axis=(3, 4))
