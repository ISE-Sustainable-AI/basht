import os

from ml_benchmark.config import MnistConfig, Path
from ml_benchmark.workload.tasks.torch_task_interface import TorchTask
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
from ml_benchmark.utils.folder_creator import FolderCreator


class MnistTorchTask(TorchTask):

    def __init__(self, config_init: dict = None) -> None:
        self.input_size = None
        self.output_size = None
        if not config_init:
            config_init = {}
        self.task_config = MnistConfig(**config_init)
        self._data_path = None
        self._data_set = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def create_dataloader(self):
        self.train_loader = DataLoader(self.train_data, batch_size=self.task_config.train_batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.task_config.val_batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.task_config.test_batch_size, shuffle=True)

    def split_data(self, val_split_ratio):
        X_train, X_val, y_train, y_val = train_test_split(
            self._data_set.train_data, self._data_set.train_labels, test_size=val_split_ratio,
            random_state=self._seed)
        self.train_set = TensorDataset(X_train, y_train)
        self.val_set = TensorDataset(X_val, y_val)
        self.test_set = TensorDataset(self._data_set.test_data, self._data_set.test_labels)

    def get_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self._data_set = MNIST(root=self._data_path, download=True, transform=transform)

    def preprocess(self):
        self._data_set.data = self._data_set.data.view(-1, self.input_size).float()

    def create_data_path(self):
        self._data_path = os.path.join(Path.data_path, "MNIST")
        FolderCreator.create_folder(self._data_path)

    def get_input_output_size(self):
        self.input_size = self._data_set.data.shape[1] * self._data_set.data.shape[2]
        self.output_size = self._data_set.targets.unique().shape[0]

    def get_loader(self):
        return self.train_loader, self.val_loader, self.test_loader


if __name__ == "__main__":
    task = MnistTask(config_init={"epochs": 1})
    task.create_data_path()
    task.get_data()
    task.get_input_output_size()
    print(task.input_size)
