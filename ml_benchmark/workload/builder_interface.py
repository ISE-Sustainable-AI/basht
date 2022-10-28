from abc import ABC, abstractmethod


class Builder(ABC):

    objective = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def build_task(self):
        pass

    @abstractmethod
    def build_model(self):
        pass
