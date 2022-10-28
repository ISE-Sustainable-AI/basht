from ml_benchmark.workload.torch_objective import TorchObjective
from ml_benchmark.workload.builder_interface import Builder


class TorchObjectiveBuilder(Builder):

    # use: https://refactoring.guru/design-patterns/builder/python/example

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.objective = TorchObjective()

    def prepare_task(self, task):
        task = task()
        task.create_data_path()
        task.get_data()
        task.get_input_output_size()
        task.preprocess()
        task.split_data()
        task.create_dataloader()
        return task

    def prepare_hyperparameter(self, model_type):
        pass

    def build_objective(self, model_type, task, hyperparameter):
        self.objective.add_task(task)
        self.objective.add_model_type(model_type)
        self.objective.add_hyperparameter(hyperparameter)


    # def create_objective(self):
    #     train_loader, val_loader, test_loader = self.create_data_loader(self.mnist_config)
    #     return self.objective_cls(
    #         self.mnist_config.epochs, train_loader, val_loader, test_loader, self.input_size,
    #         self.output_size)
