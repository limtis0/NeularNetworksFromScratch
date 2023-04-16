from src.model.layer import Layer


class Input(Layer):
    def forward(self, inputs, training=False):
        self.output = inputs

    def backward(self, d_values):
        raise NotImplementedError
