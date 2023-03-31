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
