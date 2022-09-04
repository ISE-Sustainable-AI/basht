import os

from ml_benchmark.config import MnistConfig, Path
from ml_benchmark.workload.mnist.mlp_objective import MLPObjective
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST


# TODO: config_init and hyperparameter handling have to be improved
class MnistTask:

    def __init__(self, config_init: dict = None) -> None:

        self.seed = 1337  # TODO: improve seed setting
        self.input_size = 28*28
        self.output_size = 10
        self.dataset = self._get_data()
        self.objective_cls = MLPObjective
        if not config_init:
            config_init = {}
        self.mnist_config = MnistConfig(**config_init)

    def create_data_loader(self, task_config: MnistConfig):
        train_data, val_data, test_data = self.split_data(task_config.val_split_ratio)
        train_loader = DataLoader(train_data, batch_size=task_config.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=task_config.val_batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=task_config.test_batch_size, shuffle=True)
        return train_loader, val_loader, test_loader

    def split_data(self, val_split_ratio):
        X_train, X_val, y_train, y_val = train_test_split(
            self.dataset.train_data, self.dataset.targets, test_size=val_split_ratio, random_state=self.seed)
        train_set = TensorDataset(X_train, y_train)
        val_set = TensorDataset(X_val, y_val)
        test_set = TensorDataset(self.dataset.test_data, self.dataset.test_labels)
        return train_set, val_set, test_set

    def _get_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        mnist_data = MNIST(root=os.path.join(Path.data_path, "MNIST"), download=True, transform=transform)
        mnist_data = self.mnist_preprocessing(mnist_data)
        return mnist_data

    def mnist_preprocessing(self, mnist_data):
        mnist_data.data = mnist_data.data.view(-1, 28 * 28).float()
        return mnist_data

    def create_objective(self):
        train_loader, val_loader, test_loader = self.create_data_loader(self.mnist_config)
        return self.objective_cls(
            self.mnist_config.epochs, train_loader, val_loader, test_loader, self.input_size,
            self.output_size)


if __name__ == "__main__":
    task = MnistTask(config_init={"epochs": 1})
