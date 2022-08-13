from ray import tune
import ray
from ml_benchmark.workload.mnist.mnist_task import MnistTask
import socket


grid = dict(
    input_size=28*28, learning_rate=tune.grid_search([1e-4]),
    weight_decay=1e-6,
    hidden_layer_config=tune.grid_search([[20], [10, 10]]),
    output_size=10)

#task = MnistTask(config_init={"epochs": 1})

def raytune_func(config, checkpoint_dir=None):
    import ray
    objective = ray.get(config.get("objective"))
    hyperparameters = config.get("hyperparameters")
    objective.set_hyperparameters(hyperparameters)
    # these are the results, that can be used for the hyperparameter search
    objective.train()
    validation_scores = objective.validate()
    tune.report(
        macro_f1_score=validation_scores["macro avg"]["f1-score"])

if __name__ == "__main__":
    print(socket.gethostname())
    task = MnistTask(config_init={"epochs": 1})
    objective_ref = ray.put(task.create_objective())
    analysis = tune.run(
        raytune_func,
        config=dict(
            objective=objective_ref,
            hyperparameters=grid,
        ),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
        ),
        local_dir="/home/ray/ray-results",
        resources_per_trial={"cpu": 1}
    )

    print(analysis)
