import numpy as np
from src.layer.dense import Dense
from src.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=1.0, decay=0, momentum=0):
        super().__init__(learning_rate, decay)
        self.momentum = momentum

        self.weight_momentums = {}
        self.bias_momentums = {}

    def update_params(self, layer: Dense):
        if self.momentum:
            if layer not in self.weight_momentums:
                self.weight_momentums[layer] = np.zeros_like(layer.weights)
                self.bias_momentums[layer] = np.zeros_like(layer.biases)

            # Build weight updates with momentum
            weight_updates = self.momentum * self.weight_momentums[layer] - self.current_learning_rate * layer.d_weights
            self.weight_momentums[layer] = weight_updates

            # Build bias updates with momentum
            bias_updates = self.momentum * self.bias_momentums[layer] - self.current_learning_rate * layer.d_biases
            self.bias_momentums[layer] = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.d_weights
            bias_updates = -self.current_learning_rate * layer.d_biases

        layer.weights += weight_updates
        layer.biases += bias_updates
