from ray import tune
from basht.workload.objective import Objective
from ray.tune.schedulers import HyperBandScheduler, MedianStoppingRule
from ray.tune import Stopper


class TrialStopper(Stopper):
    def __call__(self, trial_id, result):
        return result['time_total_s'] > 60*15

    def stop_all(self):
        return False


def raytune_func(config, checkpoint_dir=None):
    hyperparameter = config.get("hyperparameters")
    workload = config.get("workload")
    objective = Objective(
        dl_framework=workload.get("dl_framework"), model_cls=workload.get("model_cls"),
        epochs=workload.get("epochs"), device=workload.get("device"),
        task=workload.get("task"), hyperparameter=hyperparameter)
    # these are the results, that can be used for the hyperparameter search
    objective.load()
    objective.train()
    validation_scores = objective.validate()
    tune.report(
        macro_f1_score=validation_scores["macro avg"]["f1-score"])


def create_ray_grid(grid):
    ray_grid = {}
    for key, value in grid.items():
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, list):
            ray_grid[key] = tune.grid_search(value)
        else:
            ray_grid[key] = value
    return ray_grid


def run(workload, search_space):
    scheduler_dict = {
            "median": MedianStoppingRule,
            "hyperband": HyperBandScheduler
        }
    pruning_obj = scheduler_dict.get("median")
    """
        Executing the hyperparameter optimization on the deployed platfrom.
        use the metrics object to collect and store all measurments on the workers.
    """
    search_space = create_ray_grid(search_space)
    config = dict(
            hyperparameters=search_space,
            workload=workload
        )
    analysis = tune.run(
        raytune_func,
        config=config,
        scheduler=pruning_obj(metric="macro_f1_score", mode="max") if pruning_obj else None,
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        #local_dir="/home/ray/ray-results",
        resources_per_trial={"cpu": 2},
        stop=TrialStopper()
    )

    print(analysis.get_best_config(
        metric="macro_f1_score", mode="max")["hyperparameters"])
    return analysis


if __name__ == "__main__":
    from basht.utils.yaml import YMLHandler
    import os.path as path
    resource_definition = YMLHandler.load_yaml(path.join(path.dirname(__file__), "resource_definition.yml"))
    search_space = resource_definition["hyperparameter"]
    workload = resource_definition["workload"]
    run(workload, search_space)
