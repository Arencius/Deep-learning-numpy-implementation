import unittest
import numpy as np
from src.layers.functional import FLattenLayer
from keras.layers import Flatten


class TestFlattenLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = np.random.rand(5, 32, 32, 3)
        self.flatten = FLattenLayer()
        self.keras_flatten = Flatten()

    def test_flatten_output_shape_on_single_image(self):
        expected_output_shape = (1, 32*32*3)
        img = np.expand_dims(self.batch[0], axis=0)
        output = self.flatten.forward(img)
        self.assertEqual(expected_output_shape, output.shape)

    def test_flatten_output_shape_on_batch(self):
        expected_output_shape = (5, 32*32*3)
        output = self.flatten.forward(self.batch)
        self.assertEqual(expected_output_shape, output.shape)

    def test_flatten_output_with_keras(self):
        self.output = self.flatten.forward(self.batch)
        self.keras_output = self.keras_flatten(self.batch)
        np.testing.assert_allclose(self.output, self.keras_output)
