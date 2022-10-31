from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


class Splitter(ABC):

    _seed = 1337

    @abstractmethod
    def __init__(self, vali_split: float = None, test_split: float = None) -> None:
        pass

    @abstractmethod
    def work(self, preprocessed_dataset: Dataset):
        pass


class StandardTorchSplitter(Splitter):

    def __init__(self, vali_split: float = None, test_split: float = None) -> None:
        self.vali_split = vali_split
        self.test_split = test_split

    def work(self, preprocessed_dataset: Dataset):
        # TODO: need to add check for train and test attribute of dataset
        X_train, X_val, y_train, y_val = train_test_split(
            preprocessed_dataset.train_data, preprocessed_dataset.train_labels, test_size=self.vali_split,
            random_state=self._seed)
        trainset = TensorDataset(X_train, y_train)
        valiset = TensorDataset(X_val, y_val)
        testset = TensorDataset(self._data_set.test_data, self._data_set.test_labels)
        return trainset, valiset, testset
