from abc import ABC, abstractmethod
from matplotlib import test
from torch.utils.data import DataLoader
from basht.workload.task_components.loader import ObjDataset


class Batcher(ABC):

    @abstractmethod
    def __init__(self, train_batch_size: int, val_batch_size: int, test_batch_size: int) -> None:
        pass

    @abstractmethod
    def work(self, preprocessed_dataset: ObjDataset) -> tuple:
        pass


class TorchStandardBatcher(Batcher):

    def __init__(self, train_batch_size: int, val_batch_size: int, test_batch_size: int) -> None:
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

    def work(self, preprocessed_dataset: ObjDataset) -> tuple:
        train_loader = DataLoader(
            preprocessed_dataset.train_set, batch_size=self.train_batch_size, shuffle=True)
        val_loader = DataLoader(preprocessed_dataset.val_set, batch_size=self.val_batch_size, shuffle=True)
        test_loader = DataLoader(
            preprocessed_dataset.test_set, batch_size=self.test_batch_size, shuffle=True)
        return train_loader, val_loader, test_loader
