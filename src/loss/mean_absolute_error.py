import numpy as np
from src.loss.loss import Loss


class MeanAbsoluteError(Loss):
    def _forward(self, y_prediction: np.ndarray, y_actual: np.ndarray):
        return np.mean(np.abs(y_actual - y_prediction), axis=-1)

    def backward(self, d_values, y_actual):
        samples = len(d_values)
        labels = len(d_values[0])

        # Gradient
        self.d_inputs = np.sign(y_actual - d_values) / labels

        # Normalize
        self.d_inputs /= samples
