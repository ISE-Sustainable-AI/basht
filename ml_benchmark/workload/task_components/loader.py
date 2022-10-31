from abc import abstractmethod
import os
from utils.folder_creator import FolderCreator
from config import Path
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from typing import Tuple, List


class Loader:

    folder_creator = FolderCreator

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def work(self):
        pass


class MnistLoader(Loader):

    def __init__(self) -> None:
        self.data_path = os.path.join(Path.data_path, "MNIST")
        self.folder_creator.create_folder(self.data_path)

    def work(self) -> List[Dataset, Tuple]:
        FolderCreator.create_folder(self._data_path)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = MNIST(root=self._data_path, download=True, transform=transform)
        input_size = dataset.data.shape[1] * self._data_set.data.shape[2]
        output_size = dataset.targets.unique().shape[0]
        return [dataset, (input_size, output_size)] # TODO: Sizes need to be passed to hyperparameters
