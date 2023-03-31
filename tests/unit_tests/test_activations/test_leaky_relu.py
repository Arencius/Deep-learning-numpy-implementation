import unittest
import numpy as np
from src.activations.relu import LeakyRelu


class TestLeakyRelu(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        self.leaky_relu = LeakyRelu(0.1)
        self.forward_output = self.leaky_relu.forward(self.x)

    def test_leaky_relu_output_shape(self):
        self.assertEqual(self.forward_output.shape, self.x.shape)

    def test_leaky_relu_forward_output(self):
        expected_output = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 1, 2, 3, 4, 5])
        np.testing.assert_almost_equal(self.forward_output, expected_output)

    def test_leaky_relu_backward_output(self):
        expected_output = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_almost_equal(expected_output, self.leaky_relu.backward(self.x))
