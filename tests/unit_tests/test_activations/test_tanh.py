import unittest
import numpy as np
from src.activations.sigmoid import Tanh


class TestTanh(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-10, -5, -1, 0, 1, 5, 10])
        self.tanh = Tanh()
        self.output = self.tanh.forward(self.x)

    def test_tanh_output_shape(self):
        self.assertEqual(self.x.shape, self.output.shape)

    def test_tanh_forward_output(self):
        expected_output = np.array([
            -1.0000, -0.9999, -0.7616,  0.0000,  0.7616,  0.9999,  1.0000
        ])
        np.testing.assert_almost_equal(expected_output, self.output, decimal=5)

    def test_tanh_backward_output(self):
        expected_output = np.array([
            0.0000e+00, 1.8156e-04, 4.1997e-01, 1.0000e+00, 4.1997e-01, 1.8156e-04, 0.0000e+00
        ])
        np.testing.assert_almost_equal(expected_output, self.tanh.backward(self.x), decimal=5)

    def test_if_tanh_output_in_range(self):
        expected_min, expected_max = -1.0, 1.0
        elements_in_range = np.logical_and(
            self.output >= expected_min,
            self.output <= expected_max
        )
        self.assertTrue(np.all(elements_in_range))
