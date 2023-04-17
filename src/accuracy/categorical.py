import numpy as np
from src.accuracy.accuracy import Accuracy


class Categorical(Accuracy):
    def __init__(self):
        super().__init__()
        self.precision = None

    def initialize(self, y, reinit=False):
        pass

    def _compare(self, predictions, y):
        # Disable one-hot encoding
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y
