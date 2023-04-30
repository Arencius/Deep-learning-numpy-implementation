from abc import ABC, abstractmethod


class BaseLayer(ABC):
    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args):
        raise NotImplementedError


class Layer(BaseLayer):
    def __repr__(self):
        return f'''----------------------------------
        Name: {self.__class__.__name__}
        Parameters: {self.parameters}
        Output shape: {self.output_shape}'''

    @property
    def parameters(self):
        return 0

    @property
    def output_shape(self) -> tuple:
        raise NotImplementedError

    def _initialize_weights(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, output_gradients):
        raise NotImplementedError
