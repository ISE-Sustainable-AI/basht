from abc import ABC, abstractmethod
from sklearn.metrics import classification_report
from basht.workload.models.model_interface import ObjModel
from basht.decorators import latency_decorator, validation_latency_decorator
from numpy import random
from datetime import datetime
import torch
from basht.workload.task import TorchTask
from basht.workload.task_components import Splitter, Loader, Batcher, Preprocessor, TorchImageFlattner, \
    TorchStandardBatcher, TorchStandardSplitter, MnistLoader, FMnistLoader
from basht.workload.models import MLP
from basht.workload.objective_storage import ObjectiveStorage
from dataclasses import dataclass, field, asdict


class FunctionalObjective(ABC):

    model_cls = None
    task = None
    epochs = None
    hyperparameter = None
    device = None
    _unique_id = None
    _created_at = None
    model = None

    @abstractmethod
    def __init__(self, model_cls: ObjModel, epochs: int, device: str, hyperparameter: dict) -> None:
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def epoch_train(self):
        pass

    @abstractmethod
    def _prepare_training(self):
        pass


class ObjectiveAction:

    def __init__(self, function, *args, **kwargs) -> None:
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.function(*self.args, **self.kwargs)


class TorchObjective(FunctionalObjective):

    """
    Interface for a training, validation and test procedure of a model.
    """

    def __init__(self, model_cls: ObjModel, epochs: int, device: str, hyperparameter: dict) -> None:
        self._unique_id = random.randint(0, 1000000)
        self._created_at = datetime.now()
        self.model_cls = model_cls
        self.epochs = epochs
        self.device = torch.device(device)
        self.hyperparameter = hyperparameter

    def _prepare_training(self):
        if not self.model:
            self.model = self.model_cls(**self.hyperparameter)
            self.model = self.model.to(self.device)

    def load(self):
        self.task.prepare()
        self.hyperparameter["input_size"] = self.task.input_size
        self.hyperparameter["output_size"] = self.task.output_size

    def validate(self) -> dict:
        self.model.eval()
        if not next(self.model.parameters()).device == self.device:
            self.model = self.model.to(self.device)
        val_targets = []
        val_preds = []
        for x, y in self.task.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            predictions = self.model.test_step(x)
            targets = y.flatten().detach()
            val_targets += [targets.detach()]
            val_preds += [predictions.detach()]
        val_targets = torch.cat(val_targets).cpu().numpy()
        val_preds = torch.cat(val_preds).cpu().numpy()
        return classification_report(val_targets, val_preds, output_dict=True, zero_division=1)

    def epoch_train(self) -> list:
        if not next(self.model.parameters()).device == self.device:
            raise ValueError("Model not on objective device!")
        batch_losses = []
        for x, y in self.task.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            loss = self.model.train_step(x, y)
            batch_losses.append(loss)
        return batch_losses

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


class BuilderMapper:

    task = {
        "torch": TorchTask
    }
    torch_objects = [
        TorchImageFlattner, TorchStandardBatcher, TorchStandardSplitter, MLP, MnistLoader, FMnistLoader
    ]
    tf_objects = [
    ]
    torch_components = {cls.name: cls for cls in torch_objects}
    tf_components = {cls.name: cls for cls in tf_objects}

    components = {
        "torch": torch_components,
        "tensorflow": tf_components}

    def __init__(self, dl_framework) -> None:
        self.components = self.components.get(dl_framework)
        self.task = self.task.get(dl_framework)


class TaskBuilder:

    def __init__(self, dl_framework: str) -> None:
        self.mapper = BuilderMapper(dl_framework)
        self.task = self.mapper.task()

    def add_task_loader(self, loader: str):
        self.task.add_loader(self.mapper.components.get(loader)())

    def add_task_preprocessors(self, preprocessors: list):
        for preprocessor in preprocessors:
            preprocessor = self.mapper.components.get(preprocessor)
            self.task.add_preprocessor(preprocessor())

    def add_task_splitter(self, splitter: dict):
        splitter_cls = self.mapper.components.get(splitter.get("type"))
        splitter_config = splitter.get("config")
        self.task.add_splitter(splitter_cls(**splitter_config))

    def add_task_batcher(self, batcher: dict):
        batcher_cls = self.mapper.components.get(batcher.get("type"))
        batcher_config = batcher.get("config")
        self.task.add_batcher(batcher_cls(**batcher_config))


class Objective:

    _functional_objective = {
        "torch": TorchObjective
    }

    models = {
        MLP.name: MLP
    }

    def __init__(
            self, dl_framework: str = None, model_cls: str = None, epochs: int = 5,
            device: str = "cpu", task: dict = None, hyperparameter: dict = None) -> None:
        self.workload_definition = locals()
        if hyperparameter:
            self.hyperparameter = hyperparameter
        else:
            self.hyperparameter = {}
        self.model_cls = self.models.get(model_cls)
        self.epochs = epochs
        self.device = device
        self._functional_objective = self._functional_objective.get(
            dl_framework)(self.model_cls, self.epochs, self.device, self.hyperparameter)
        self._functional_objective.task = self._build_task(dl_framework, task)
        self.objective_storage = ObjectiveStorage()

    def _build_task(self, dl_framework: str, task_definition: dict) -> any:
        task_builder = TaskBuilder(dl_framework)
        if task_definition:
            task_builder.add_task_loader(task_definition.get("loader"))
            task_builder.add_task_preprocessors(task_definition.get("preprocessors"))
            task_builder.add_task_splitter(task_definition.get("splitter"))
            task_builder.add_task_batcher(task_definition.get("batcher"))
            print("Finished Task Building")
        else:
            raise AttributeError("No propper task definition provided")
        return task_builder.task

    @latency_decorator
    def load(self):
        return self._functional_objective.load()

    @latency_decorator
    def train(self, objective_action: callable = None, with_validation: bool = False):
        self._functional_objective._prepare_training()
        for epoch in range(1, self.epochs + 1):
            self.objective_storage.current_epoch = epoch
            batch_losses = self._functional_objective.epoch_train()
            # TODO: loss summarization should be custom
            self.objective_storage.add_training_scores(sum(batch_losses)/len(batch_losses))
            if with_validation:
                validation_results = self._functional_objective.validate()
                self.objective_storage.add_validation_scores(validation_results)
            if objective_action:
                objective_action()
        return self.objective_storage.get_current_epoch_results()

    @validation_latency_decorator
    def validate(self):
        return self._functional_objective.validate()

    def test(self):
        return self._functional_objective.test()
