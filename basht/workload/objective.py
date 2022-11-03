from abc import ABC, abstractmethod

from basht.workload.objective_director import ObjectiveDirector


class ObjectiveInterface(ABC):

    workload_definition = None

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

    def __init__(
            self, dl_framework: str = None, model_cls: str = None, epochs: int = 5,
            device: str = "cpu", task: dict = None, hyperparameter: dict = None) -> None:
        self.workload_definition = locals()
        self._director = ObjectiveDirector()
        self._functional_objective = self.director.build_objective(self.workload_definition)

    @classmethod
    def from_graph(cls, workload_config_graph):
        pass

    def train(self):
        return self._functional_objective.train()

    def validate(self):
        return self._functional_objective.validate()

    def test(self):
        return self._functional_objective.test()
