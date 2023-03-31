import numpy as np


class Softmax:
    def __init__(self):
        self.output = []

    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        