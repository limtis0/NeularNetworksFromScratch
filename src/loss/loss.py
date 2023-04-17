import numpy as np
from abc import ABC, abstractmethod
from src.layer.dense import Dense


class Loss(ABC):
    def __init__(self):
        self.d_inputs = None

        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, output: np.ndarray, y_actual: np.ndarray):
        sample_losses = self._forward(output, y_actual)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        return np.mean(sample_losses)

    def calculate_accumulated(self):
        return self.accumulated_sum / self.accumulated_count

    def reset_accumulated(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    @staticmethod
    def get_regularization_loss(*layers: Dense):
        regularization_loss = 0

        for layer in layers:
            if layer.l1_regularizer_w > 0:
                regularization_loss += layer.l1_regularizer_w * np.sum(np.abs(layer.weights))

            if layer.l2_regularizer_w > 0:
                regularization_loss += layer.l2_regularizer_w * np.sum(layer.weights * layer.weights)

            if layer.l1_regularizer_b > 0:
                regularization_loss += layer.l1_regularizer_b * np.sum(np.abs(layer.biases))

            if layer.l2_regularizer_b > 0:
                regularization_loss += layer.l2_regularizer_b * np.sum(layer.biases * layer.biases)

        return regularization_loss

    @abstractmethod
    def _forward(self, y_prediction: np.ndarray, y_actual: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def backward(self, d_values, y_actual):
        raise NotImplementedError
