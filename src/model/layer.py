from abc import ABC, abstractmethod
from typing import Optional


class Layer(ABC):
    def __init__(self):
        self.prev: Optional[Layer] = None
        self.next: Optional[Layer] = None

        self.inputs = None
        self.output = None

        self.d_inputs = None

    @abstractmethod
    def forward(self, inputs, training=False):
        raise NotImplementedError

    @abstractmethod
    def backward(self, d_values):
        raise NotImplementedError
