import unittest
import numpy as np
from src.activations.sigmoid import Softmax


class TestSoftmax(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-10, -5, -1, 0, 1, 5, 10])
        self.softmax = Softmax()
        self.output = self.softmax.forward(self.x)

    def test_softmax_output_shape(self):
        self.assertEqual(self.x.shape, self.output.shape)

    def test_softmax_output(self):
        expected_output = np.array([
            2.0470e-09, 3.0380e-07, 1.6587e-05, 4.5088e-05,
            1.2256e-04, 6.6916e-03, 9.9312e-01
        ])
        np.testing.assert_almost_equal(expected_output, self.output, decimal=5)
