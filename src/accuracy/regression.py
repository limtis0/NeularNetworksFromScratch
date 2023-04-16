import numpy as np
from src.accuracy.accuracy import Accuracy


class Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def initialize(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def _compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
