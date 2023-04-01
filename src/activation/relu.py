import numpy as np


class ReLU:
    def __init__(self):
        self.output = None
        self.inputs = None

        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0
