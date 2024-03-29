import os
import sys

import optuna

from basht.utils.generate_grid_search_space import generate_grid_search_space
from basht.utils.yaml import YMLHandler
from basht.workload.objective import Objective, ObjectiveAction
from basht.workload.objective_storage import ObjectiveStorageInterface


class OptunaTrial:

    def __init__(self, search_space, dl_framework, model_cls, epochs, device, task, pruning) -> None:
        self.search_space = search_space
        self.dl_framework = dl_framework
        self.model_cls = model_cls
        self.epochs = epochs
        self.device = device
        self.task = task
        self.pruning = pruning

    def __call__(self, trial):
        if self.search_space.get("hidden_lyer_config"):
            hidden_layer_idx = trial.suggest_categorical(
                "hidden_layer_config", list(self.search_space["hidden_layer_config"].keys()))
            hidden_layer_config = self.search_space.get("hidden_layer_config")[hidden_layer_idx]
        else:
            hidden_layer_config = None
        if self.search_space.get("learning_rate"):
            lr = trial.suggest_float(
                "learning_rate", min(self.search_space["learning_rate"]),
                max(self.search_space["learning_rate"]), log=True)
        else:
            lr = None
        if self.search_space.get("weight_decay"):
            decay = trial.suggest_float(
                "weight_decay", min(self.search_space["weight_decay"]),
                max(self.search_space["weight_decay"]), log=True)
        else:
            decay = None
        hyperparameter = {
            "learning_rate": lr, "weight_decay": decay,
            "hidden_layer_config": hidden_layer_config
        }
        hyperparameter = {key: value for key, value in hyperparameter.items() if value}

        self.objective = Objective(
            dl_framework=self.dl_framework, model_cls=self.model_cls, epochs=self.epochs, device=self.device,
            task=self.task, hyperparameter=hyperparameter)
        self.objective.load()
        if self.pruning or not isinstance(self.pruning, optuna.pruners.NopPruner):
            # TODO: no pruning should not require a validation - this has to be tested again
            objective_storage_interface = ObjectiveStorageInterface(self.objective)
            objective_action = ObjectiveAction(
                OptunaTrial.pruning_function, trial=trial,
                objective_storage_interface=objective_storage_interface)
            results = self.objective.train(objective_action=objective_action, with_validation=True)[1]
        else:
            self.objective.train()
            results = self.objective.validate()
        return results["macro avg"]["f1-score"]

    @staticmethod
    def pruning_function(trial, objective_storage_interface):
        validation_scores = objective_storage_interface.get_validation_scores()[-1]["macro avg"]["f1-score"]
        epoch = objective_storage_interface.get_current_epoch()
        trial.report(validation_scores, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()


def main():
    pruning_dict = {
        "median": optuna.pruners.MedianPruner(),
        "hyperband": optuna.pruners.HyperbandPruner(),
        None: optuna.pruners.NopPruner()
    }

    try:
        resource_path = os.path.join(os.path.dirname(__file__), "resource_definition.yml")
        resource_def = YMLHandler.load_yaml(resource_path)
        print(resource_def)
        study_name = resource_def.get("studyName")
        database_conn = os.environ.get("DB_CONN")
        hyperparameter = resource_def.get("hyperparameter")
        pruning_obj = pruning_dict.get(resource_def.get("pruning"))
        print(pruning_obj)

        search_space = generate_grid_search_space(hyperparameter)
        workload_def = resource_def.get("workload")
        optuna_trial = OptunaTrial(
            search_space, dl_framework=workload_def.get("dl_framework"),
            model_cls=workload_def.get("model_cls"),
            epochs=workload_def.get("epochs"), device=workload_def.get("device"),
            task=workload_def.get("task"), pruning=resource_def.get("pruning"))
        study = optuna.create_study(
            study_name=study_name, storage=database_conn, direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.GridSampler(search_space), pruner=pruning_obj)
        study.optimize(optuna_trial)

        # this used to be important but nobody rembered why ... sleep(5)
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
