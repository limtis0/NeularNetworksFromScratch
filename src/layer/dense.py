import numpy as np


class Dense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

        self.inputs = None
        self.output = None

        # Back-propagation
        self.d_inputs = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        self.d_inputs = np.dot(d_values, self.weights.T)
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
