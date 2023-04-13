import numpy as np
from src.activation.softmax import Softmax
from src.loss.categorical_crossentropy import CategoricalCrossentropy


# Combined softmax activation and categorical cross-entropy loss
# For faster backward step
class SoftmaxClassifier:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()

        self.output = None
        self.d_inputs = None

    def forward(self, inputs, y_actual):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_actual)

    def backward(self, d_values, y_actual):
        samples = len(d_values)

        if len(y_actual.shape) == 2:
            y_actual = np.argmax(y_actual, axis=1)

        self.d_inputs = d_values.copy()
        self.d_inputs[range(samples), y_actual] -= 1
        self.d_inputs /= samples
