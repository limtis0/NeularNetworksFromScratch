import numpy as np


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
        self.output = []

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
