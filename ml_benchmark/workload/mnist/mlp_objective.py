import torch
import tqdm
from ml_benchmark.config import MLPHyperparameter
from ml_benchmark.decorators import latency_decorator, validation_latency_decorator
from ml_benchmark.workload.mnist.mlp import MLP
from ml_benchmark.workload.objective import Objective
from sklearn.metrics import classification_report


# TODO: use a task for initialization?
class MLPObjective(Objective):

    def __init__(self, epochs, train_loader, val_loader, test_loader, input_size, output_size) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.val_laoder = val_loader
        self.test_loader = test_loader
        self.hyperparameters = MLPHyperparameter().to_dict()
        self.epochs = epochs
        self.device = None
        self.input_size = input_size
        self.output_size = output_size

    def set_hyperparameters(self, hyperparameters: dict):
        hyperparameters["input_size"] = self.input_size
        hyperparameters["output_size"] = self.output_size
        print(self.hyperparameters)
        self.hyperparameters.update(hyperparameters)
        print(self.hyperparameters)
    
    def get_hyperparameters(self) -> dict:
        return self.hyperparameters

    def set_device(self, device_str: str = None):
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cpu")

    @latency_decorator
    def train(self):
        # model setup
        self.model = MLP(**self.hyperparameters)
        self.model = self.model.to(self.device)
        # train
        epoch_losses = []
        for epoch in tqdm.tqdm(range(1, self.epochs+1)):
            batch_losses = []
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.model.train_step(x, y)
                batch_losses.append(loss)
            epoch_losses.append(sum(batch_losses)/len(batch_losses))
        return {"train_loss": epoch_losses}

    @validation_latency_decorator
    def validate(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        val_targets = []
        val_preds = []
        for x, y in self.val_laoder:
            x = x.to(self.device)
            y = y.to(self.device)
            predictions = self.model.test_step(x)
            targets = y.flatten().detach()
            val_targets += [targets.detach()]
            val_preds += [predictions.detach()]
        val_targets = torch.cat(val_targets).cpu().numpy()
        val_preds = torch.cat(val_preds).cpu().numpy()
        return classification_report(val_targets, val_preds, output_dict=True, zero_division=1)

    def test(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        test_targets = []
        test_predictions = []
        for x, y in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            predictions = self.model.test_step(x)
            targets = y.flatten().detach()
            test_targets += [targets.detach()]
            test_predictions += [predictions.detach()]
        test_targets = torch.cat(test_targets).cpu().numpy()
        test_predictions = torch.cat(test_predictions).cpu().numpy()
        return classification_report(test_targets, test_predictions, output_dict=True, zero_division=1)
