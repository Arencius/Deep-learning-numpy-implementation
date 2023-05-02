import unittest
import numpy as np
from src.optimizers.optimizers import SGD


class TestSGD(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_params = np.ones((10, 25))
        self.dummy_grads = np.zeros_like(self.dummy_params) + 0.1
        self.sgd = SGD()

    def test_sgd_output_with_zero_momentum(self):
        expected_output_params = np.full_like(self.dummy_params, 0.999)
        self.sgd.update_params(self.dummy_params, self.dummy_grads)

        np.testing.assert_array_equal(expected_output_params, self.dummy_params)

    def test_sgd_output_with_momentum(self):
        params = np.ones((10, 25))
        self.sgd_momentum = SGD(momentum=0.05)

        expected_output_params = np.full_like(params, 0.99905)
        self.sgd_momentum.update_params(params, self.dummy_grads)

        np.testing.assert_array_equal(expected_output_params, params)
