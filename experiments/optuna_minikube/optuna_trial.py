import os
import sys
from time import sleep
import optuna
from basht.workload.objective import Objective
from basht.utils.yaml import YMLHandler
from utils import generate_search_space
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


class OptunaTrial:

    def __init__(self, search_space, dl_framework, model_cls, epochs, device, task) -> None:
        self.search_space = search_space
        self.dl_framework = dl_framework
        self.model_cls = model_cls
        self.epochs = epochs
        self.device = device
        self.task = task

    def __call__(self, trial):
        # TODO: optuna does not take lists for gridsearch and sampling -
        # you need to add building of lists internally
        hidden_layer_idx = trial.suggest_categorical(
            "hidden_layer_config", list(self.search_space["hidden_layer_config"].keys()))
        lr = trial.suggest_float(
            "learning_rate", self.search_space["learning_rate"].min(), self.search_space["learning_rate"].max(), log=True)
        decay = trial.suggest_float(
            "weight_decay", self.search_space["weight_decay"].min(), self.search_space["weight_decay"].max(), log=True)
        hyperparameter = {
            "learning_rate": lr, "weight_decay": decay,
            "hidden_layer_config": self.search_space.get("hidden_layer_config")[hidden_layer_idx]
        }

        self.objective = Objective(
            dl_framework=self.dl_framework, model_cls=self.model_cls, epochs=self.epochs, device=self.device,
            task=self.task, hyperparameter=hyperparameter)
        self.objective.load()
        for epoch in range(1, self.epochs+1):
            self.objective.epoch_train()
            validation_scores = self.objective.validate()
            trial.report(validation_scores["macro avg"]["f1-score"], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return validation_scores["macro avg"]["f1-score"]


def main():
    try:
        resource_path = os.path.join(os.path.dirname(__file__), "resource_definition.yml")
        resource_def = YMLHandler.load_yaml(resource_path)
        print(resource_def)
        study_name = os.environ.get("STUDY_NAME", "Test-Study")
        database_conn = os.environ.get("DB_CONN")
        n_trials = resource_def.get("trials")
        hyperparameter = resource_def.get("hyperparameter")
        search_space = generate_search_space(hyperparameter)
        workload_def = resource_def.get("workload")
        optuna_trial = OptunaTrial(
            search_space, dl_framework=workload_def.get("dl_framework"),
            model_cls=workload_def.get("model_cls"),
            epochs=workload_def.get("epochs"), device=workload_def.get("device"),
            task=workload_def.get("task"))
        study = optuna.create_study(
            study_name=study_name, storage=database_conn, direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.GridSampler(search_space), pruner=optuna.pruners.MedianPruner())
        study.optimize(
            optuna_trial,
            callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))])
        sleep(5)
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
