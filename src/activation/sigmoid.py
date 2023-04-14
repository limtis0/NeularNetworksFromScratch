import numpy as np
from src.abstract.layer import Layer


class Sigmoid(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, d_values):
        self.d_inputs = d_values * (1 - self.output) * self.output
