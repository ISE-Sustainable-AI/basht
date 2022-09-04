from abc import ABC, abstractmethod
from numpy import random
from datetime import datetime

from ml_benchmark.config import MetricsStorageConfig


class Objective(ABC):

    """
    Interface for a training, validation and test procedure of a model.
    """

    def __init__(self) -> None:
        self._unique_id = random.randint(0, 1000000)
        self._created_at = datetime.now()

    @abstractmethod
    def set_hyperparameters(self, hyperparameters: dict):
        """
        Set the hyperparameters of the objective.
        """
        pass

    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """
        Get the hyperparameters of the objective.
        """
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass
