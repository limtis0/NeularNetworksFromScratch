from src.layer.dense import Dense


class SGD:
    def __init__(self, learning_rate=1.0, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learing_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        # LR decay
        if self.decay:
            self.current_learing_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer: Dense):
        if self.momentum:
            # Build weight updates with momentum
            weight_updates = self.momentum * layer.weight_momentums - self.current_learing_rate * layer.d_weights
            layer.weight_momentums = weight_updates

            # Build bias updates with momentum
            bias_updates = self.momentum * layer.bias_momentums - self.current_learing_rate * layer.d_biases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learing_rate * layer.d_weights
            bias_updates = -self.current_learing_rate * layer.d_biases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1
