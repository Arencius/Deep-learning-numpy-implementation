from abc import ABC, abstractmethod


class BaseLayer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args):
        raise NotImplementedError
