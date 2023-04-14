import numpy as np
from src.loss.loss import Loss


class BinaryCrossentropy(Loss):
    def _forward(self, y_prediction: np.ndarray, y_actual: np.ndarray):
        # Prevent DivByZero error
        y_pred_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)

        sample_losses = -(y_actual * np.log(y_pred_clipped) + (1 - y_actual) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, d_values, y_actual):
        samples = len(d_values)
        labels = len(d_values[0])

        # Prevent DivByZero error
        d_values_clipped = np.clip(d_values, 1e-7, 1 - 1e-7)

        # Gradient
        self.d_inputs = -(y_actual / d_values_clipped - (1 - y_actual) / (1 - d_values_clipped)) / labels

        # Normalize
        self.d_inputs = self.d_inputs / samples
