import unittest
import numpy as np
from src.activations.relu import LeakyRelu


class TestLeakyRelu(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        self.leaky_relu = LeakyRelu(0.1)
        self.output = self.leaky_relu.forward(self.x)

    def test_leaky_relu_output_shape(self):
        self.assertEqual(self.output.shape, self.x.shape)

    def test_leaky_relu_output(self):
        expected_output = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 1, 2, 3, 4, 5])
        np.testing.assert_almost_equal(self.output, expected_output)
