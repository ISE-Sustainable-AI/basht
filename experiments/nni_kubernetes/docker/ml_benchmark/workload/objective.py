from abc import ABC, abstractmethod
import os

from ml_benchmark.config import MetricsStorageConfig


class Objective(ABC):

    """
    Interface for a training, validation and test procedure of a model.
    """

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass
