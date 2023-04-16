import numpy as np
from src.model.layer import Layer
from src.activation.activation import Activation


class Sigmoid(Layer, Activation):
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, d_values):
        self.d_inputs = d_values * (1 - self.output) * self.output

    def get_predictions(self, output):
        return (output > 0.5) * 1
