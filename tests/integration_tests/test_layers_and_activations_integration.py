import unittest
import numpy as np
from src.layers.dense import DenseLayer
from src.layers.convolution import Conv2DLayer
from src.layers.pooling import MaxPoolingLayer
from src.layers.functional import FLattenLayer
from src.activations.sigmoid import Softmax
from src.activations.relu import Relu


class TestLayersAndActivations(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = np.random.randint(0, 255,
                                       (5, 32, 32, 3))
        self.conv = Conv2DLayer(input_size=self.batch.shape[1:],
                                output_filters=64,
                                padding=True)
        self.pool = MaxPoolingLayer(input_size=(32, 32, 64),
                                    pool_size=2)
        self.flatten = FLattenLayer()
        self.dense = DenseLayer(input_neurons=16 * 16 * 64,
                                output_neurons=10)
        self.relu = Relu()
        self.softmax = Softmax()

        self.pipeline = [self.conv, self.pool, self.relu, self.flatten, self.dense, self.softmax]

    def _run_pipeline(self):
        x = self.batch
        for layer in self.pipeline:
            x = layer.forward(x)
        return x

    def test_if_pipeline_runs(self):
        self._run_pipeline()

    def test_pipeline_output_shape(self):
        expected_output_shape = (5, 10)
        output = self._run_pipeline()

        self.assertEqual(expected_output_shape, output.shape)
