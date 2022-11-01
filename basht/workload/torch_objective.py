from numpy import random
from datetime import datetime
import tqdm
import torch
from sklearn.metrics import classification_report

from basht.workload.task import TorchTask
from basht.workload.models.model_interface import ObjModel
from basht.decorators import latency_decorator, validation_latency_decorator


class TorchObjective:

    """
    Interface for a training, validation and test procedure of a model.
    """

    def __init__(self) -> None:
        self._unique_id = random.randint(0, 1000000)
        self._created_at = datetime.now()
        self.model_cls = None
        self.hyperparameters = None
        self.task = None
        self.device = None

    @latency_decorator
    def train(self) -> dict:
        # prepare data
        self.task.prepare()

        # model setup
        self.model = self.model_cls(**self.hyperparameters)
        self.model = self.model.to(self.device)
        # train
        epoch_losses = []
        for epoch in tqdm.tqdm(range(1, self.epochs+1)):
            batch_losses = []
            for x, y in self.task.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                loss = self.model.train_step(x, y)
                batch_losses.append(loss)
            epoch_losses.append(sum(batch_losses)/len(batch_losses))
        return {"train_loss": epoch_losses}

    @validation_latency_decorator
    def validate(self) -> dict:
        self.model.eval()
        self.model = self.model.to(self.device)
        val_targets = []
        val_preds = []
        for x, y in self.task.val_laoder:
            x = x.to(self.device)
            y = y.to(self.device)
            predictions = self.model.test_step(x)
            targets = y.flatten().detach()
            val_targets += [targets.detach()]
            val_preds += [predictions.detach()]
        val_targets = torch.cat(val_targets).cpu().numpy()
        val_preds = torch.cat(val_preds).cpu().numpy()
        return classification_report(val_targets, val_preds, output_dict=True, zero_division=1)

    def test(self) -> dict:
        self.model.eval()
        self.model = self.model.to(self.device)
        test_targets = []
        test_predictions = []
        for x, y in self.task.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            predictions = self.model.test_step(x)
            targets = y.flatten().detach()
            test_targets += [targets.detach()]
            test_predictions += [predictions.detach()]
        test_targets = torch.cat(test_targets).cpu().numpy()
        test_predictions = torch.cat(test_predictions).cpu().numpy()
        return classification_report(test_targets, test_predictions, output_dict=True, zero_division=1)

    def _add_task(self, task: TorchTask):
        self.task = task

    def _add_model_cls(self, model_cls: ObjModel):
        self.model_cls = model_cls

    def set_device(self, device: str):
        self.device = device

    def set_hyperparameters(self, hyperparameters: dict):
        self.hyperparameters = hyperparameters
