from abc import ABC, abstractmethod


class Builder(ABC):

    task = None
    model = None
    objective = None

    @abstractmethod
    def __init__(self, task, model_type) -> None:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def build_task(self):
        pass

    @abstractmethod
    def build_model(self):
        pass
