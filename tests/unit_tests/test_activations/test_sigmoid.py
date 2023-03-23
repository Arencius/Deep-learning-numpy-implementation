import unittest
import numpy as np
from src.activations.sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-10, -5, -1, 0, 1, 5, 10])
        self.sigmoid = Sigmoid()
        self.output = self.sigmoid.forward(self.x)

    def test_sigmoid_output_shape(self):
        self.assertEqual(self.x.shape, self.output.shape)

    def test_sigmoid_output(self):
        expected_output = np.array([
            4.53978687e-05, 6.69285092e-03, 2.68941421e-01,
            5.00000000e-01, 7.31058579e-01, 9.93307149e-01, 9.99954602e-01
        ])
        np.testing.assert_almost_equal(expected_output, self.output)
