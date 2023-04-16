import numpy as np
from src.model.layer import Layer
from src.activation.activation import Activation


class ReLU(Layer, Activation):
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0

    def get_predictions(self, output):
        return output
