import os
import sys
from time import sleep
import optuna
from basht.workload.mnist.mnist_task import MnistTask
from utils import generate_search_space
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


class OptunaTrial:

    def __init__(self, search_space) -> None:
        self.search_space = search_space

    def __call__(self, trial):
        epochs = int(os.environ.get("EPOCHS", 5))
        task = MnistTask(config_init={"epochs": epochs})
        objective = task.create_objective()
        lr = trial.suggest_float(
            "learning_rate", self.search_space["learning_rate"].min(), self.search_space["learning_rate"].max(), log=True)
        decay = trial.suggest_float(
            "weight_decay", self.search_space["weight_decay"].min(), self.search_space["weight_decay"].max(), log=True)
        # TODO: optuna does not take lists for gridsearch and sampling -
        # you need to add building of lists internally
        # hidden_layer_config = trial.suggest_categorical(
        #     "hidden_layer_config", self.search_space["hidden_layer_config"])
        objective.set_hyperparameters(
            {"learning_rate": lr, "weight_decay": decay})
        objective.train()
        validation_scores = objective.validate()
        return validation_scores["macro avg"]["f1-score"]


def main():
    try:
        study_name = os.environ.get("STUDY_NAME", "Test-Study")
        database_conn = os.environ.get("DB_CONN")
        n_trials = int(os.environ.get("N_TRIALS", 2))
        search_space = generate_search_space(
            os.path.join(os.path.dirname(__file__), "hyperparameter_space.yml"))
        search_space
        optuna_trial = OptunaTrial(search_space)
        study = optuna.create_study(
            study_name=study_name, storage=database_conn, direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.GridSampler(search_space))
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
