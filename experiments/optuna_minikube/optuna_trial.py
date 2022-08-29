import os
import sys
from time import sleep
import optuna
from ml_benchmark.workload.mnist.mnist_task import MnistTask
from utils import generate_search_space


def optuna_trial(trial):
    task = MnistTask(config_init={"epochs": 10})
    objective = task.create_objective()
    # optuna doesnt care, these lines of code just get hyperparameters from the search space in grid search
    lr = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
    decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    hidden_layer_config = trial.suggest_int("hidden_layer_config", 1, 4)
    objective.set_hyperparameters(
        {"learning_rate": lr, "weight_decay": decay, "hidden_layer_config": hidden_layer_config})
    objective.train()
    validation_scores = objective.validate()
    return validation_scores["macro avg"]["f1-score"]


if __name__ == "__main__":
    try:
        study_name = os.environ.get("STUDY_NAME")
        database_conn = os.environ.get("DB_CONN")
        search_space = generate_search_space("hyp_space_definition.yml")
        study = optuna.create_study(
            study_name=study_name, storage=database_conn, direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(optuna_trial)
        # TODO: add small wait to avoid missing metrics
        sys.exit(0)
    except Exception:
        sys.exit(1)
