import unittest
import numpy as np
from src.activations.sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-10, -5, -1, 0, 1, 5, 10])
        self.sigmoid = Sigmoid()
        self.forward_output = self.sigmoid.forward(self.x)

    def test_sigmoid_output_shape(self):
        self.assertEqual(self.x.shape, self.forward_output.shape)

    def test_sigmoid_forward_output(self):
        expected_output = np.array([
            4.53978687e-05, 6.69285092e-03, 2.68941421e-01,
            5.00000000e-01, 7.31058579e-01, 9.93307149e-01, 9.99954602e-01
        ])
        np.testing.assert_almost_equal(expected_output, self.forward_output)

    def test_sigmoid_backward_output(self):
        expected_output = np.array([
            4.5396e-05, 6.6481e-03, 1.9661e-01, 2.5000e-01, 1.9661e-01, 6.6480e-03,
            4.5417e-05
        ])
        np.testing.assert_almost_equal(self.sigmoid.backward(self.x), expected_output, decimal=5)
