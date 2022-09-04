import multiprocessing
import time
from itertools import product
from multiprocessing import cpu_count
import os
import json

ctx = multiprocessing.get_context("spawn")


class VanillaGridSearch:
    def __init__(
        self, objective_cls, objective_args, grid, result_path=None, resources=None) -> None:
        """_summary_

        Args:
            objective (_type_): _description_
            grid (dict): Need to be a dict of iterables.
            device (str): pytorch device string
            num_processes (int, optional): Defaults to number of availible CPUS. Number of num_processes to use.
        """
        self.objective_cls = objective_cls
        self.objective_args = objective_args
        self.grid = self._generate_parameters(grid)
        self.device = device
        if resources:
            self.num_processes = resources.get("num_processes", cpu_count())
            self.device = resources.get("device", "cpu")
        else:
            self.num_processes = cpu_count()
            self.deivce = "cpu"

        if result_path:
            self.result_path = result_path
        else:
            self.result_path = os.getcwd()

    def _generate_parameters(self, grid):
        return product([dict(zip(grid, v)) for v in product(*grid.values())])

    def _run_objective(self, hyperparameter_config):
        job = TuneJob(self.objective_cls, self.objective_args, self.device, hyperparameter_config[0])
        score = job.main()
        return {'params': hyperparameter_config[0], 'score': score}

    def main(self):
        start = time.time()
        grid_search_score_dump = []
        if self.num_processes > 1:
            grid_search_score_dump = self._multiprocess_run(
                self.objective_cls, self.objective_args, self.device, self.num_processes)
            # with ctx.Pool(self.num_processes) as p:
            #     grid_search_score_dump = p.map(self._run_objective, self.grid)
        else:
            for hyper_config in self.grid:
                grid_search_score_dump.append(self._run_objective(hyper_config))

        self._save_grid_search_results(grid_search_score_dump)
        print(f'Grid search took {time.time() - start} seconds')

    def _save_grid_search_results(self, results):
        with open(os.path.join(self.result_path, "grid_search_results.json"), "w") as f:
            json.dump(results, f)
        print(f"Saved Results to: {self.result_path}")

    def _multiprocess_run(self, objective_cls, objective_args, device, num_processes):
        with ctx.Manager() as manager:
            scores = manager.list()
            queue = ctx.Queue()
            processes = []
            for hyper_config in self.grid:
                queue.put(hyper_config)

            for _ in range(num_processes):
                p = TuneProcess(queue, objective_cls, objective_args, device, scores)
                processes.append(p)
                p.start()

            for p in processes:
                p.join()
                p.terminate()
            queue.close()
            scores = scores[:]
        return scores


class TuneProcess(ctx.Process):
    def __init__(self, queue, objective_cls, objective_args, device, scores):
        super().__init__()
        self.queue = queue
        self.objective_cls = objective_cls
        self.objective_args = objective_args
        self.device = device
        self.scores = scores

    def run(self):
        while not self.queue.empty():
            hyperparameter_config = self.queue.get()
            job = TuneJob(self.objective_cls, self.objective_args, self.device, hyperparameter_config[0])
            score = job.main()
            self.scores.append(score)


class TuneJob:
    def __init__(self, objective_cls, objective_args, device, hyperparameters) -> None:
        self.objective = objective_cls(**objective_args)
        self.device = device
        self.hyperparameters = hyperparameters

    def main(self):
        self.objective.set_hyperparameters(self.hyperparameters)
        self.objective.set_device(self.device)
        self.objective.train()
        return self.objective.validate()["macro avg"]["f1-score"]


if __name__ == "__main__":
    from ml_benchmark.workload.mnist.mlp_objective import MLPObjective
    from ml_benchmark.workload.mnist.mnist_task import MnistTask
    epochs = 5
    configuration = {
            "val_split_ratio": 0.2, "train_batch_size": 512, "val_batch_size": 128, "test_batch_size": 128}
    task = MnistTask()
    grid = dict(
            input_size=[task.input_size],
            learning_rate=[1e-4, 1e-2],
            weight_decay=[1e-6],
            hidden_layer_config=[[20], [10, 10]],
            output_size=[task.output_size])
    resources = dict(processes=2)

    train_loader, val_loader, test_loader = task.create_data_loader(configuration)
    objective_cls = MLPObjective
    objective_args = dict(
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
        )
    device = "cpu"
    grid_search = VanillaGridSearch(
        objective_cls=objective_cls, objective_args=objective_args, grid=grid, device=device,
        num_processes=resources["processes"])
    grid_search.main()
