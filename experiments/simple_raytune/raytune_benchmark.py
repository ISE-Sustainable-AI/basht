from ray import tune

from ml_benchmark.benchmark_runner import Benchmark, BenchmarkRunner
from ml_benchmark.config import MnistConfig
from ml_benchmark.latency_tracker import Latency, LatencyTracker


class RaytuneBenchmark(Benchmark):

    def __init__(self, objective, grid, resources) -> None:
        self.objective = objective
        self.grid = grid
        self.resources = resources

    def deploy(self) -> None:
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """
        pass

    def setup(self):
        pass

    def run(self):
        """
            Executing the hyperparameter optimization on the deployed platfrom.
            use the metrics object to collect and store all measurments on the workers.
        """
        def raytune_func(config):
            """The function for training and validation, that is used for hyperparameter optimization.
            Beware Ray Synchronisation: https://docs.ray.io/en/latest/tune/user-guide.html

            Args:
                config ([type]): [description]
            """
            objective = config.get("objective")
            hyperparameters = config.get("hyperparameters")
            objective.set_hyperparameters(hyperparameters)
            # these are the results, that can be used for the hyperparameter search
            with Latency(objective.train) as latency:
                objective.train()
            latency_tracker = LatencyTracker(address="")
            latency_tracker.track(latency)
            validation_scores = objective.validate()
            tune.report(macro_f1_score=validation_scores["macro avg"]["f1-score"])

        self.analysis = tune.run(
            raytune_func,
            config=dict(
                objective=self.objective,
                hyperparameters=self.grid,
                ),
            sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
                ),
            local_dir="/tmp/ray_results",
            resources_per_trial={"cpu": self.resources["cpu"]}
        )

    def collect_run_results(self):
        self.best_config = self.analysis.get_best_config(metric="macro_f1_score", mode="max")["hyperparameters"]
        self.latency = self.analysis.dataframe(metric="latency")

    def test(self):
        # evaluating and retrieving the best model to generate test results.
        self.objective.set_hyperparameters(self.best_config)
        self.training_loss = self.objective.train()
        self.test_scores = self.objective.test()

    def collect_benchmark_metrics(self):
        """
            Describes the collection of all gathered metrics, which are not used by the HPO framework
            (Latencies, CPU Resources, etc.). This step runs outside of the HPO Framework.
            Ensure to optain all metrics loggs and combine into the metrics object.
        """
        results = dict(
            test_scores=self.test_scores,
            training_loss=self.training_loss,
            latency=self.latency
            )

        return results

    def undeploy(self):
        """
            The clean-up procedure to undeploy all components of the HPO Framework that were deployed in the
            Deploy step.
        """
        pass


if __name__ == "__main__":

    # The basic config for the workload. For testing purposes set epochs to one.
    # For benchmarking take the default value of 100
    config = MnistConfig(epochs=1).to_dict()
    # your ressources the optimization should run on
    resources = {"cpu": 12}
    # Add your hyperparameter setting procedure here
    # your hyperparameter grid you want to search over
    hyperparameters = dict(
            input_size=28*28, learning_rate=tune.grid_search([1e-4]),
            weight_decay=1e-6,
            hidden_layer_config=tune.grid_search([[20], [10, 10]]),
            output_size=10)

    # import an use the runner
    runner = BenchmarkRunner(
        benchmark_cls=RaytuneBenchmark, config=config, grid=hyperparameters, resources=resources,
        task_str="mnist")
    runner.run()
