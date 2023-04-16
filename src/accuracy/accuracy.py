import numpy as np
from abc import ABC, abstractmethod


class Accuracy(ABC):
    def calculate(self, predictions, y):
        comparisons = self._compare(predictions, y)
        return np.mean(comparisons)

    @abstractmethod
    def initialize(self, y, reinit=False):
        raise NotImplementedError

    @abstractmethod
    def _compare(self, predictions, y):
        raise NotImplementedError
