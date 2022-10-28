from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def train_step(self, x, y):
        pass

    @abstractmethod
    def test_step(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass
