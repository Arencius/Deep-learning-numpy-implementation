from src.layers.layer import BaseLayer

import unittest


class TestBaseLayer(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_if_layer_implements_forward_method(self):
        self.assertTrue(hasattr(BaseLayer, 'forward'))

    def test_if_layer_implements_backward_method(self):
        self.assertTrue(hasattr(BaseLayer, 'backward'))
