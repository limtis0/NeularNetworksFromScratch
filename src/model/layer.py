from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.inputs = None
        self.output = None

        self.d_inputs = None

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, d_values):
        raise NotImplementedError
