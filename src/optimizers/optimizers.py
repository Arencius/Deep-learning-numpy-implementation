from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update_params(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate=1e-2, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, params, grads):
        grads = grads * (1.0 - self.momentum)

        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad


class Adam(Optimizer):
    def __init__(self, learning_rate=1e-3, alpha=0.9, beta=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.alpha = alpha
        self.beta = beta

        self.t = 0
        self.running_mean = None
        self.running_var = None

    def update_params(self, params, grads):
        grads = grads.reshape(params.shape)

        if self.running_mean is None:
            self.running_mean = np.zeros_like(grads)
            self.running_var = np.zeros_like(grads)

        self.t += 1
        self.running_mean = self.running_mean * self.alpha + (1.0 - self.alpha) * grads
        self.running_var = self.running_var * self.beta + (1.0 - self.beta) * (grads ** 2)

        rm = self.running_mean / (1.0 - self.alpha ** self.t)
        rv = self.running_var / (1.0 - self.beta ** self.t)

        for param, grad in zip(params, grads):
            param -= self.learning_rate * rm / (np.sqrt(rv) + self.eps)
