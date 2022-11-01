from abc import ABC, abstractmethod
from matplotlib import test
from torch.utils.data import DataLoader
from typing import Tuple
from dataclasses import dataclass


@dataclass
class LoaderTuple:

    train_loader: DataLoader = None
    vali_loader: DataLoader = None
    test_loader: DataLoader = None


class Batcher(ABC):

    @abstractmethod
    def __init__(self, train_batch_size: int, vali_batch_size: int, test_batch_size: int) -> None:
        pass

    @abstractmethod
    def work(self):
        pass


class TorchBatcher(Batcher):

    def __init__(self, train_batch_size: int, vali_batch_size: int, test_batch_size: int) -> None:
        self.train_batch_size = train_batch_size
        self.vali_batch_size = vali_batch_size
        self.test_batch_size = test_batch_size

    def work(self) -> LoaderTuple:
        train_loader = DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=True)
        return LoaderTuple(train_loader, val_loader, test_loader)
