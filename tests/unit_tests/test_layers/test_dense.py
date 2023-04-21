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

    def test_forward_output_shape(self):
        expected_shape = (1, self.output_neurons)
        self.assertEqual(expected_shape, self.dense.output_shape)

    def test_number_of_parameters(self):
        expected_no_params = (128 * 256) + 256
        self.assertEqual(expected_no_params, self.dense.parameters)

    def test_forward_output_shape_on_batch(self):
        batch = np.random.rand(5, 128)
        expected_output_shape = (5, 256)
        output = self.dense.forward(batch)
        self.assertEqual(expected_output_shape, output.shape)