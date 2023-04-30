import unittest
import numpy as np
from src.layers.functional import DropoutLayer


class TestDropoutLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = np.random.randint(1, 1000, size=(5, 256,))
        self.dropout = DropoutLayer(p=0.5)

    def test_dropout_output_shape(self):
        output = self.dropout.forward(self.batch)
        self.assertEqual(self.batch.shape, output.shape)
