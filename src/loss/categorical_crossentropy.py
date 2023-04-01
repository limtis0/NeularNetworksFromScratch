import numpy as np
from src.loss.loss import Loss


class CategoricalCrossentropy(Loss):
    def _forward(self, y_prediction, y_actual):
        samples = len(y_actual)
        clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)

        # If not one-hot encoded
        if len(y_actual.shape) == 1:
            correct_confidences = clipped[range(samples), y_actual]
        else:
            correct_confidences = np.sum(y_prediction * y_actual, axis=1)

        return -np.log(correct_confidences)

    def backward(self, d_values, y_actual):
        samples = len(d_values)
        labels = len(d_values[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_actual.shape) == 1:
            y_actual = np.eye(labels)[y_actual]

        # Calculate gradient
        self.d_inputs = -y_actual / d_values

        # Normalize gradient
        self.d_inputs /= samples
