import unittest
import numpy as np
from src.activations.sigmoid import Softmax


class TestSoftmax(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-10, -5, -1, 0, 1, 5, 10])
        self.softmax = Softmax()
        self.forward_output = self.softmax.forward(self.x)

    def test_softmax_output_shape(self):
        self.assertEqual(self.x.shape, self.forward_output.shape)

    def test_softmax_forward_output(self):
        expected_output = np.array([
            2.0470e-09, 3.0380e-07, 1.6587e-05, 4.5088e-05,
            1.2256e-04, 6.6916e-03, 9.9312e-01
        ])
        np.testing.assert_almost_equal(expected_output, self.forward_output, decimal=5)

    def test_softmax_backward_output(self):
        expected_output = np.array([
            2.0470e-09, 3.0380e-07, 1.6587e-05, 4.5086e-05, 1.2255e-04, 6.6468e-03,
            6.8289e-03
        ])
        np.testing.assert_almost_equal(expected_output, self.softmax.backward(self.x), decimal=5)

    def test_softmax_forward_output_range(self):
        expected_min, expected_max = 0.0, 1.0
        elements_in_range = np.logical_and(
            self.forward_output >= expected_min,
            self.forward_output <= expected_max
        )
        self.assertTrue(np.all(elements_in_range))

    def test_if_softmax_output_sums_to_1(self):
        self.assertEqual(self.forward_output.sum(), 1.0)
