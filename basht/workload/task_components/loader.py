from abc import abstractmethod
import os
from basht.utils.folder_creator import FolderCreator
from basht.config import Path
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset


class ObjDataset:

    def __init__(self, dataset: Dataset, input_size: int = None, output_size: int = None) -> None:
        self._input_size = input_size
        self._output_size = output_size
        self.dataset = dataset
        self.train_set = None
        self.val_set = None
        self.test_set = None
        # TODO: add logic for targets, train set, test set, etc.

        @property
        def input_size(self) -> int:
            if self._input_size:
                return self._input_size
            else:
                raise Warning("No input size has been set for your dataset. Workload will not be able to \
                    proceed")

        @input_size.setter
        def input_size(self, input_size: int):
            self._input_size = input_size

        @property
        def output_size(self) -> int:
            if self._output_size:
                return self._output_size
            else:
                raise Warning("No output size has been set for your dataset. Workload will not be able \
                            to proceed")

        @output_size.setter
        def output_size(self, output_size: int):
            self._output_size = output_size


class Loader:

    folder_creator = FolderCreator

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def work(self) -> ObjDataset:
        pass


class MnistLoader(Loader):

    def __init__(self) -> None:
        self.data_path = os.path.join(Path.data_path, "MNIST")
        self.folder_creator.create_folder(self.data_path)

    def work(self) -> ObjDataset:
        FolderCreator.create_folder(self.data_path)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = MNIST(root=self.data_path, download=True, transform=transform)
        return ObjDataset(dataset)
