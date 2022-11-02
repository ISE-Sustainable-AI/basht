from abc import ABC, abstractmethod

from basht.workload.task_components.loader import ObjDataset


class Preprocessor(ABC):

    name = None

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def work(self, dataset: ObjDataset) -> ObjDataset:
        pass


class TorchImageFlattner(Preprocessor):

    name = "ImageFlattner"

    def __init__(self) -> None:
        pass

    def work(self, dataset: ObjDataset) -> ObjDataset:
        # TODO: addjust for more generality
        dataset.input_size = dataset.dataset.data.shape[1] * dataset.dataset.data.shape[2]
        dataset.output_size = dataset.dataset.targets.unique().shape[0]
        dataset.dataset.data = dataset.dataset.data.view(-1, dataset.input_size).float()
        return dataset
