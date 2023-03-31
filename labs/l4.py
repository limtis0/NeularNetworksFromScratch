import numpy as np


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
        self.output = []

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Using input batches
def ex_1():
    # Inputs
    inputs = np.array([
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ])

    # Layer
    weights = np.array([
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ])
    biases = [2, 3, 0.5]

    # Output
    output = np.dot(inputs, weights.T) + biases

    print(output)


# Using multiple layers
def ex_2():
    # Inputs
    inputs = np.array([
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ])

    # Layer 1
    weights1 = np.array([
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ])
    biases1 = [2, 3, 0.5]

    # Layer 2
    weights2 = np.array([
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13],
    ])
    biases2 = [-1, 2, -0.5]

    # Output
    layer1_output = np.dot(inputs, weights1.T) + biases1
    layer2_output = np.dot(layer1_output, weights2.T) + biases2

    print(layer2_output)


def ex_3():
    np.random.seed(0)

    X = np.array([
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ])

    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 2)

    layer1.forward(X)
    # print(layer1.output)

    layer2.forward(layer1.output)
    print(layer2.output)


if __name__ == '__main__':
    ex_3()
