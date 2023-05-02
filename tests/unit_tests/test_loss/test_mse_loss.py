import numpy as np
import unittest
from src.loss.loss_functions import MeanSquaredErrorLoss


class TestMeanSquaredErrorLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.y_true = np.array([2.0, 4.5, 1.0, 7.0, 5.5])
        self.y_pred = np.array([2.5, 4.0, 1.5, 6.5, 6.0])
        self.mse = MeanSquaredErrorLoss()

    def test_mse_forward_output_on_single_inputs(self):
        loss = self.mse.forward(self.y_true, self.y_pred)
        self.assertEqual(0.25, loss)

    def test_mse_forward_output_on_batch(self):
        y_true = np.repeat(np.expand_dims(self.y_true, 0), 5, axis=0)
        y_pred = np.repeat(np.expand_dims(self.y_pred, 0), 5, axis=0)

        loss = self.mse.forward(y_true, y_pred)
        self.assertEqual(0.25, loss)

    def test_mse_forward_output_with_same_inputs(self):
        self.assertEqual(0.0, self.mse.forward(self.y_true, self.y_true))

    def test_mse_backward_output(self):
        expected_output = np.array([0.5, -0.5, 0.5, -0.5, 0.5])
        gradient = self.mse.backward(self.y_true, self.y_pred)
        np.testing.assert_array_equal(expected_output, gradient)
