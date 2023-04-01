import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self):
        self.d_inputs = None

    def calculate(self, output: np.ndarray, y_actual: np.ndarray):
        sample_losses = self._forward(output, y_actual)
        return np.mean(sample_losses)

    @abstractmethod
    def _forward(self, y_prediction: np.ndarray, y_actual: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def backward(self, d_values, y_actual):
        raise NotImplementedError
