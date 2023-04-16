from src.model.layer import Layer
from src.activation.activation import Activation


class Linear(Layer, Activation):
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = inputs

    def backward(self, d_values):
        self.d_inputs = d_values.copy()

    def get_predictions(self, output):
        return output
