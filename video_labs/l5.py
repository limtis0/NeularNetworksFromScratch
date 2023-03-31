import nnfs
from nnfs.datasets import spiral_data
import numpy as np
from l4 import LayerDense


class ActivationReLU:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Basic ReLU activation
def ex_1():
    inputs = [0.2, -1, 3.3, -2.7, 1.1, 2.2, -100]
    output = []

    for i in inputs:
        output.append(max(0, i))

    print(output)


def ex_2():
    X, y = spiral_data(100, 3)
    layer1 = LayerDense(2, 5)
    activation1 = ActivationReLU()

    layer1.forward(X)
    activation1.forward(layer1.output)
    print(activation1.output)


if __name__ == '__main__':
    nnfs.init()
    ex_2()
