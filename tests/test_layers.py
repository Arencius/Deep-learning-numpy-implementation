from src.layers import Layer

import unittest


class TestLayer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_if_layer_implements_forward_method(self):
        self.assertTrue(hasattr(Layer, 'forward'))

    def test_if_layer_implements_backward_method(self):
        self.assertTrue(hasattr(Layer, 'backward'))
