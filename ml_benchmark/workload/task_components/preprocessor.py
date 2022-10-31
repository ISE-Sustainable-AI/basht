from abc import ABC, abstractmethod


class Preprocessor(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def work(self, dataset):
        pass


class TorchImageFlattner(Preprocessor):

    def __init__(self) -> None:
        pass

    def work(self, dataset):
        dataset.data = self.dataset.data.view(-1, self.input_size).float()
        return dataset
