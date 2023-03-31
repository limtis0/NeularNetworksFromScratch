import nnfs
from nnfs.datasets import spiral_data
import numpy as np
from l4 import LayerDense
from l5 import ActivationReLU


class ActivationSoftmax:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)


# Exponentiating values to get rid of negatives
def ex_1():
    layer_outputs = [4.8, 1.21, 2.385]
    print(np.exp(layer_outputs))


# Normalizing exp values
def ex_2():
    layer_outputs = [4.8, 1.21, 2.385]
    exp = np.exp(layer_outputs)
    norm = exp / np.sum(exp)
    print(norm)
    print(sum(norm))


# Same but with batches
def ex_3():
    layer_outputs = [
        [4.8, 1.21, 2.385],
        [8.9, -1.81, 0.2],
        [1.41, 1.051, 0.026]
    ]
    exp = np.exp(layer_outputs)
    norm = exp / np.sum(exp, axis=1, keepdims=True)
    print(norm)


def ex_4():
    X, y = spiral_data(samples=100, classes=3)

    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])


if __name__ == '__main__':
    nnfs.init()
    ex_4()
