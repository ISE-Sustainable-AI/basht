from ml_benchmark.workload.mnist_task import MnistTask
from ml_benchmark.workload.mlp_objective import MLPObjective
from ml_benchmark.utils.result_saver import ResultSaver
from ray import tune
import time
import os


# -------------  Only runs if you install raytune: https://github.com/ray-project/ray -----------------------
def raytune_func(config):
    """The function for training and validation, that is used for hyperparameter optimization.
    Beware Ray Synchronisation: https://docs.ray.io/en/latest/tune/user-guide.html

    Args:
        config ([type]): [description]
    """
    objective = config.get("objective")
    hyperparameters = config.get("hyperparameters")
    device = config.get("device")
    objective.set_device(device)
    objective.set_hyperparameters(hyperparameters)
    # these are the results, that can be used for the hyperparameter search
    objective.train()
    validation_scores = objective.validate()
    tune.report(macro_f1_score=validation_scores["macro avg"]["f1-score"])


def main(epochs, configuration, hyperparameters, device=None):
    task = MnistTask()
    train_loader, val_loader, test_loader = task.create_data_loader(configuration)
    objective = MLPObjective(
        epochs=epochs, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    # starting the tuning
    start_time = time.time()  # we want to measure the time from start of the hyperparameter search to generation of test results
    analysis = tune.run(
        raytune_func,
        config=dict(
            objective=objective,
            hyperparameters=hyperparameters,
            device=device
            ),
        sync_config=tune.SyncConfig(
            syncer=None  # Disable syncing
            ),
        local_dir="/tmp/ray_results",
        resources_per_trial={"cpu": 12, "gpu": 1 if device else 0}
        )

    # evaluating and retrieving the best model to generate test results.
    objective.set_device(device)
    objective.set_hyperparameters(analysis.get_best_config(
            metric="macro_f1_score", mode="max")["hyperparameters"])
    training_loss = objective.train()
    test_scores = objective.test()

    # these are the results needed for the benchmark. Please make sure to always generate test results!
    results = dict(
        execution_time=time.time() - start_time,
        test_scores=test_scores,
        training_loss=training_loss
    )
    saver = ResultSaver(
        experiment_name="GridSearch_MLP_MNIST", experiment_path=os.path.abspath(os.path.dirname(__file__)))
    saver.save_results(results)


if __name__ == "__main__":
    device = "cpu"  # GPUs can only be enabled via setting in raytune
    # configuration for your job and data, please do not change!
    epochs = 50
    configuration = {
            "val_split_ratio": 0.2, "train_batch_size": 512, "val_batch_size": 128, "test_batch_size": 128}
    # Add your hyperparameter setting procedure here
    # your hyperparameter grid you want to search over
    hyperparameters = dict(
            input_size=28*28, learning_rate=tune.grid_search([1e-4, 1e-2]),
            weight_decay=1e-6,
            hidden_layer_config=tune.grid_search([[20], [10, 10]]),
            output_size=10)
    main(epochs, configuration, hyperparameters, device)