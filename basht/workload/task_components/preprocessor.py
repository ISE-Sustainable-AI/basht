from abc import ABC, abstractmethod

from basht.workload.task_components.loader import ObjDataset


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

    def work(self, dataset: ObjDataset):
        # TODO: addjust for more generality
        dataset.input_size = dataset.data.shape[1] * self._data_set.data.shape[2]
        dataset.output_size = dataset.targets.unique().shape[0]
        dataset.dataset.data = self.dataset.data.view(-1, dataset.input_size).float()
        return dataset
