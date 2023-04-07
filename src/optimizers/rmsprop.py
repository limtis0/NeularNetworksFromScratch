import numpy as np
from src.optimizers.optimizer import Optimizer
from src.layer.dense import Dense


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, rho=0.9):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho

        self.weight_cache = {}
        self.bias_cache = {}

    def update_params(self, layer: Dense):
        if layer not in self.weight_cache:
            self.weight_cache[layer] = np.zeros_like(layer.weights)
            self.bias_cache[layer] = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        self.weight_cache[layer] = self.rho * self.weight_cache[layer] + (1 - self.rho) * layer.d_weights ** 2
        self.bias_cache[layer] = self.rho * self.bias_cache[layer] + (1 - self.rho) * layer.d_biases ** 2

        # Vanilla SGD parameter update + normalization
        # but with square rooted cache
        layer.weights += -self.current_learning_rate * layer.d_weights \
            / (np.sqrt(self.weight_cache[layer] + self.epsilon))

        layer.biases += -self.current_learning_rate * layer.d_biases \
            / (np.sqrt(self.bias_cache[layer] + self.epsilon))
