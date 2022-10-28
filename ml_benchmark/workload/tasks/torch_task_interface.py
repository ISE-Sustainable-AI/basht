from abc import ABC, abstractmethod


class TorchTask(ABC):

    def get_data(self):
        pass

    @abstractmethod
    def create_dataloader(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def split_data(self):
        pass
