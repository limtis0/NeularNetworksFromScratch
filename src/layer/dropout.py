import numpy as np
from src.model.layer import Layer


class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()

        self.rate = 1 - rate
        self.binary_mask = None

    def forward(self, inputs, training=False):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=self.inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, d_values):
        self.d_inputs = d_values * self.binary_mask
