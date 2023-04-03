import unittest
import numpy as np
from src.layers.pooling import AveragePoolingLayer
from keras.layers import AveragePooling2D


class TestAveragePoolingLayer(unittest.TestCase):
    """
    Since the only difference between MaxPool and AveragePool is the operation (all output size are the same)
    the only test here is the output result.
    """

    def setUp(self) -> None:
        self.height, self.width, self.channels = (64, 64, 3)
        self.pool_size = 2
        self.stride = 2
        self.batch_size = 5
        self.images = np.random.normal(size=(self.batch_size, self.height, self.width, self.channels))

        self.avg_pool = AveragePoolingLayer(input_size=(self.height, self.width, self.channels),
                                            pool_size=self.pool_size,
                                            stride=self.stride)

        self.keras_avg_pool = AveragePooling2D(pool_size=self.pool_size,
                                               strides=self.stride)

    def test_average_pool_forward_output_with_one_image(self):
        img = np.expand_dims(self.images[2], 0)

        pool_output = self.avg_pool.forward(img)
        keras_output = self.keras_avg_pool(img)

        np.testing.assert_almost_equal(pool_output, keras_output, decimal=5)

    def test_average_pool_forward_output_with_batch(self):
        batch = self.images

        pool_output = self.avg_pool.forward(batch)
        keras_output = self.keras_avg_pool(batch)

        np.testing.assert_almost_equal(pool_output, keras_output, decimal=5)
