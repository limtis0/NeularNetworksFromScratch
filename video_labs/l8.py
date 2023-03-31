import nnfs
import numpy as np
from abc import ABC, abstractmethod
from nnfs.datasets import spiral_data
from l4 import LayerDense
from l5 import ActivationReLU
from l6 import ActivationSoftmax


class Loss(ABC):
    def calculate(self, output: np.ndarray, y_actual: np.ndarray):
        sample_losses = self._forward(output, y_actual)
        return np.mean(sample_losses)

    @abstractmethod
    def _forward(self, y_prediction: np.ndarray, y_actual: np.ndarray):
        raise NotImplementedError


class CategoricalCrossentropyLoss(Loss):
    def _forward(self, y_prediction, y_actual):
        samples = len(y_actual)
        clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)

        # If not one-hot encoded
        if len(y_actual.shape) == 1:
            correct_confidences = clipped[range(samples), y_actual]
        # If one-hot encoded
        else:
            correct_confidences = np.sum(y_prediction * y_actual, axis=1)

        return -np.log(correct_confidences)


# Calculating average loss (not accounting 0)
def ex_1():
    softmax_outputs = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.02, 0.9, 0.08]
    ])
    class_target = [0, 1, 1]

    loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_target])
    avg_loss = np.mean(loss)
    print(avg_loss)


def ex_2():
    X, y = spiral_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_func = CategoricalCrossentropyLoss()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_func.calculate(activation2.output, y)
    print(loss)


if __name__ == '__main__':
    nnfs.init()
    ex_2()
