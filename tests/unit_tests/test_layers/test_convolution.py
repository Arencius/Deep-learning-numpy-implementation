import unittest
import numpy as np
from src.layers.convolution import Conv2DLayer
from keras.layers import Conv2D


class TestConv2DLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.height, self.width, self.channels = (64, 64, 3)
        self.output_filters = 128
        self.batch_size = 5
        self.image = np.random.rand(1, self.height, self.width, self.channels)
        self.batch = np.random.rand(self.batch_size, self.height, self.width, self.channels)

        self.conv = Conv2DLayer(input_size=(self.height, self.width, self.channels),
                                output_filters=self.output_filters,
                                padding=True)

        self.keras_conv = Conv2D(self.output_filters,
                                 kernel_size=3,
                                 input_shape=(1, self.height, self.width, self.channels),
                                 weights=[self.conv.filters],
                                 use_bias=False,    # since biases are initialized as zeros it won't make any difference
                                 padding='same',
                                 strides=1)

    def test_if_implements_forward_method(self):
        self.assertTrue(hasattr(Conv2DLayer, 'forward'))

    def test_if_implements_backward_method(self):
        self.assertTrue(hasattr(Conv2DLayer, 'backward'))

    def test_filters_shape(self):
        expected_shape = (self.channels, 3, 3, self.output_filters)
        self.assertEqual(expected_shape, self.conv.filters.shape)

    def test_number_of_parameters(self):
        expected_no_params = (self.channels * 128 * 3 * 3) + 128  # (channels,  output_filters, filter_size, filter_size) + bias
        self.assertEqual(expected_no_params, self.conv.parameters)

    def test_conv_output_shape(self):
        expected_output_shape = (self.height, self.width, 128)
        self.assertEqual(expected_output_shape, self.conv.output_shape)

    def test_padding_output_shape_on_single_image(self):
        expected_shape = (1, self.height + 2, self.width + 2, self.channels)
        padded = self.conv.pad_image(self.image)
        self.assertEqual(expected_shape, padded.shape)

    def test_padding_output_shape_on_batch(self):
        expected_shape = (self.batch_size, self.height + 2, self.width + 2, self.channels)
        padded = self.conv.pad_image(self.batch)
        self.assertEqual(expected_shape, padded.shape)

    def test_conv_output_on_single_image(self):
        output = self.conv.forward(self.image)
        keras_output = self.keras_conv(self.image)

        np.testing.assert_almost_equal(output, keras_output, decimal=4)

    def test_conv_output_on_batch(self):
        output = self.conv.forward(self.batch)
        keras_output = self.keras_conv(self.batch)

        np.testing.assert_almost_equal(output, keras_output, decimal=4)
