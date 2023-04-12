import numpy as np


class Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

        self.inputs = None
        self.binary_mask = None
        self.output = None
        self.d_inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, d_values):
        self.d_inputs = d_values * self.binary_mask
