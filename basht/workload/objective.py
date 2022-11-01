
from typing import function
from abc import ABC, abstractmethod

from basht.workload.objective_director import ObjectiveDirector


class ObjectiveInterface(ABC):

    hyperparameters = None
    device = None
    pruning_callback = None

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass


class Objective(ObjectiveInterface):

    def __init__(self, workload_definition: dict) -> None:
        self._director = ObjectiveDirector(workload_definition)
        self._functional_objective = self.director.build_objective()

    def set_device(self, device: str):
        self._functional_objective.device = device

    def set_hyperparameters(self, hyperparameters: dict):
        self._functional_objective.hyperparameters = hyperparameters

    def set_pruning_callback(self, callback_method: function):
        pass

    def train(self):
        return self._functional_objective.train()

    def validate(self):
        return self._functional_objective.validate()

    def test(self):
        return self._functional_objective.test()
