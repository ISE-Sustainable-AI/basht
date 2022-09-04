import os
import sys
from time import sleep
import optuna
from ml_benchmark.workload.mnist.mnist_task import MnistTask
from utils import generate_search_space
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState



def optuna_trial(trial):
    epochs = int(os.environ.get("EPOCHS", 5))
    task = MnistTask(config_init={"epochs": epochs})
    objective = task.create_objective()
    # optuna doesnt care, these lines of code just get hyperparameters from the search space in grid search
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    # hidden_layer_config = trial.suggest_int("hidden_layer_config", 1, 4)
    objective.set_hyperparameters(
        {"learning_rate": lr, "weight_decay": decay})#, "hidden_layer_config": hidden_layer_config})
    objective.train()
    validation_scores = objective.validate()
    return validation_scores["macro avg"]["f1-score"]

def main():
    try:
        study_name = os.environ.get("STUDY_NAME", "Test-Study")
        database_conn = os.environ.get("DB_CONN")
        n_trials = int(os.environ.get("N_TRIALS", 2))
        search_space = generate_search_space(os.path.join(os.path.dirname(__file__),"hyperparameter_space.yml"))
        print(search_space)
        study = optuna.create_study(
            study_name=study_name, storage=database_conn, direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(optuna_trial,
            callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,))],
        ) ##TODO:XXX We need to make this a configurable parameter!!!
        # TODO: add small wait to avoid missing metrics
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
