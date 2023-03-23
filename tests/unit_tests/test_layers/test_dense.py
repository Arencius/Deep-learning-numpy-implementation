import unittest
import numpy as np
from src.layers.dense import DenseLayer


class TestDenseLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.input_neurons, self.output_neurons = 128, 256
        self.x = np.random.randint(0, 256, size=self.input_neurons)
        self.dense = DenseLayer(self.input_neurons,
                                self.output_neurons)

    def test_if_implements_forward_method(self):
        self.assertTrue(hasattr(DenseLayer, 'forward'))

    def test_if_implements_backward_method(self):
        self.assertTrue(hasattr(DenseLayer, 'backward'))

    def test_weights_shape(self):
        expected_shape = (self.input_neurons, self.output_neurons)
        self.assertEqual(expected_shape, self.dense.weights.shape)

    def test_output_shape(self):
        expected_shape = (1, self.output_neurons)
        output_shape = self.dense.forward(self.x).shape
        self.assertEqual(expected_shape, output_shape)