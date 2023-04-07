from src.layer.dense import Dense
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        # LR decay
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    @abstractmethod
    def update_params(self, layer: Dense):
        raise NotImplementedError

    def post_update_params(self):
        self.iterations += 1
