import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    def accuracy(self, y_true, y_pred):
        return 0

    @abstractmethod
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    def accuracy(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        return np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)) / batch_size

    def forward(self, y_true, y_pred):
        eps = 1e-12  # to avoid division by zero
        output = -np.sum(y_true * np.log(y_pred + eps), axis=1)
        return np.mean(output)

    def backward(self, y_true, y_pred):
        eps = 1e-12  # to avoid division by zero
        return - (y_true / y_pred + eps) + (1 - y_true) / (1 - y_pred + eps)


class BinaryCrossEntropyLoss(Loss):
    def accuracy(self, y_true, y_pred):
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return np.mean(y_pred == y_true)

    def forward(self, y_true, y_pred):
        eps = 1e-8  # small constant to avoid division by zero
        return -np.mean(y_true * np.log(y_pred + eps) + (1.0 - y_true) * np.log(1.0 - y_pred + eps))

    def backward(self, y_true, y_pred):
        eps = 1e-8  # small constant to avoid division by zero
        return (y_pred - y_true) / (y_pred * (1.0 - y_pred) + eps)


class MeanSquaredErrorLoss(Loss):
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):
        return -(y_true - y_pred)
