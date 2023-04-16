import numpy as np
from src.abstract.layer import Layer


class Linear(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        