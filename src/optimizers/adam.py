import numpy as np
from src.optimizers.optimizer import Optimizer
from src.layer.dense import Dense


class Adam(Optimizer):
    # Adam is a combination of RMSProp and momentum
    # Beta_2 hyperparameter is basically an RMSProp's rho
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.weight_momentums = {}
        self.bias_momentums = {}
        self.weight_cache = {}
        self.bias_cache = {}

    def update_params(self, layer: Dense):
        if layer not in self.weight_momentums:
            self.weight_momentums[layer] = np.zeros_like(layer.weights)
            self.bias_momentums[layer] = np.zeros_like(layer.biases)
            self.weight_cache[layer] = np.zeros_like(layer.weights)
            self.bias_cache[layer] = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        self.weight_momentums[layer] = self.beta_1 * self.weight_momentums[layer] + (1 - self.beta_1) * layer.d_weights
        self.bias_momentums[layer] = self.beta_1 * self.bias_momentums[layer] + (1 - self.beta_1) * layer.d_biases

        # Get corrected momentum
        weight_momentums_corrected = self.weight_momentums[layer] / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = self.bias_momentums[layer] / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        self.weight_cache[layer] = self.beta_2 * self.weight_cache[layer] + (1 - self.beta_2) * layer.d_weights ** 2
        self.bias_cache[layer] = self.beta_2 * self.bias_cache[layer] + (1 - self.beta_2) * layer.d_biases ** 2

        # Get corrected cache
        weight_cache_corrected = self.weight_cache[layer] / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = self.bias_cache[layer] / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # but with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected \
            / (np.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums_corrected \
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
