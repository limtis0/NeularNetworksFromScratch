import numpy as np
from abc import ABC, abstractmethod


class Accuracy(ABC):
    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, predictions, y):
        comparisons = self._compare(predictions, y)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return np.mean(comparisons)

    def calculate_accumulated(self):
        return self.accumulated_sum / self.accumulated_count

    def reset_accumulated(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    @abstractmethod
    def initialize(self, y, reinit=False):
        raise NotImplementedError

    @abstractmethod
    def _compare(self, predictions, y):
        raise NotImplementedError
