import numpy as np
from src.model.layer import Layer
from src.activation.activation import Activation


class Softmax(Layer, Activation):
    def forward(self, inputs, training=False):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, d_values):
        self.d_inputs = np.empty_like(d_values)

        # Enumerate outputs and gradients
        for i, (single_output, single_d_values) in enumerate(zip(self.output, d_values)):
            single_output = single_output.reshape(-1, 1)  # Flatten
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.d_inputs[i] = np.dot(jacobian_matrix, single_d_values)

    def get_predictions(self, output):
        return np.argmax(output, axis=1)
