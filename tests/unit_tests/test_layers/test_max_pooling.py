import unittest
import numpy as np
from src.layers.pooling import MaxPoolingLayer
from keras.layers import MaxPool2D


class TestMaxPoolingLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.height, self.width, self.channels = (64, 64, 3)
        self.pool_size = 2
        self.stride = 2
        self.batch_size = 5
        self.images = np.random.randint(0, 256, (self.batch_size, self.height, self.width, self.channels))

        self.maxpool = MaxPoolingLayer(input_size=(self.height, self.width, self.channels),
                                       pool_size=self.pool_size,
                                       stride=self.stride)

        self.keras_maxpool = MaxPool2D(pool_size=self.pool_size,
                                       strides=self.stride)

    def test_if_implements_forward_method(self):
        self.assertTrue(hasattr(MaxPoolingLayer, 'forward'))

    def test_if_implements_backward_method(self):
        self.assertTrue(hasattr(MaxPoolingLayer, 'backward'))

    def test_output_shape_with_stride_2(self):
        img = np.expand_dims(self.images[0], 0)
        keras_pool_output = self.keras_maxpool(img)

        expected_output_shape = keras_pool_output.shape[1:]  # remove the batch dimension
        self.assertEqual(expected_output_shape, self.maxpool.output_shape)

    def test_output_shape_with_stride_1(self):
        self.maxpool.stride = 1
        keras_maxpool_stride_1 = MaxPool2D(strides=1)

        img = np.expand_dims(self.images[1], 0)
        keras_pool_output = keras_maxpool_stride_1(img)

        expected_output_shape = keras_pool_output.shape[1:]
        self.assertEqual(expected_output_shape, self.maxpool.output_shape)

    def test_max_pool_forward_output_with_one_image(self):
        img = np.expand_dims(self.images[2], 0)

        pool_output = self.maxpool.forward(img)
        keras_output = self.keras_maxpool(img)

        np.testing.assert_array_equal(pool_output, keras_output)

    def test_max_pool_forward_output_with_batch(self):
        batch = self.images

        pool_output = self.maxpool.forward(batch)
        keras_output = self.keras_maxpool(batch)

        np.testing.assert_array_equal(pool_output, keras_output)
