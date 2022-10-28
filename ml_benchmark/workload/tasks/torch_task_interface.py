from abc import ABC, abstractmethod


class TorchTask(ABC):

    seed = 1337

    @abstractmethod
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

    @abstractmethod
    def get_input_output_size(self):
        pass

    @abstractmethod
    def create_data_path(self):
        pass

    @abstractmethod
    def get_loader(self):
        pass
