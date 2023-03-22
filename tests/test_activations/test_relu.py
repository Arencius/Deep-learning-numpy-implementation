import unittest
import numpy as np
from src.activations.relu import Relu


class TestRelu(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        self.relu = Relu()
        self.output = self.relu.forward(self.x)

    def test_output_shape(self):
        self.assertEqual(self.output.shape, self.x.shape)

    def test_if_output_is_correct(self):
        expected_output = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(self.output, expected_output)
