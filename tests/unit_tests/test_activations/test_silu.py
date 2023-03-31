import unittest
import numpy as np
from src.activations.sigmoid import SiLU


class TestSiLU(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-10, -5, -1, 0, 1, 5, 10])
        self.silu = SiLU()
        self.output = self.silu.forward(self.x)

    def test_silu_output_shape(self):
        self.assertEqual(self.x.shape, self.output.shape)

    def test_silu_output(self):
        expected_output = np.array([
            -4.5398e-04, -3.3464e-02, -2.6894e-01, 0.0000e+00, 7.3106e-01,
            4.9665e+00, 9.9995e+00
        ])
        np.testing.assert_almost_equal(expected_output, self.output, decimal=4)
