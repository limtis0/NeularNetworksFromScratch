import numpy as np
from src.abstract.layer import Layer


class Dense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int, l1_regularizer_w: float = 0, l2_regularizer_w: float = 0,
                 l1_regularizer_b: float = 0, l2_regularizer_b: float = 0):
        super().__init__()

        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

        # Back-propagation
        self.d_weights = None
        self.d_biases = None

        # L1 and L2 regularization strength
        self.l1_regularizer_w = l1_regularizer_w
        self.l2_regularizer_w = l2_regularizer_w
        self.l1_regularizer_b = l1_regularizer_b
        self.l2_regularizer_b = l2_regularizer_b

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values):
        # Gradient on parameters
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        # Gradient on regularization
        if self.l1_regularizer_w > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self.d_weights += self.l1_regularizer_w * d_l1

        if self.l2_regularizer_w > 0:
            self.d_weights += 2 * self.l2_regularizer_w * self.weights

        if self.l1_regularizer_b > 0:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases < 0] = -1
            self.d_biases += self.l1_regularizer_b * d_l1

        if self.l2_regularizer_b > 0:
            self.d_biases += 2 * self.l2_regularizer_b * self.biases

        # Gradient on values
        self.d_inputs = np.dot(d_values, self.weights.T)
