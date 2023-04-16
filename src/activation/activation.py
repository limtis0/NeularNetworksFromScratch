from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def get_predictions(self, output):
        raise NotImplementedError
