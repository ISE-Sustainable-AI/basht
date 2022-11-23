import nni
from ml_benchmark.workload.mnist.mnist_task import MnistTask

hyperparammeters = nni.get_next_parameter()
task = MnistTask(config_init={"epochs": 100})
objective = task.create_objective()
objective.set_hyperparameters(hyperparammeters)

objective.train()
validation_scores = objective.validate()

nni.report_final_result(validation_scores["macro avg"]["f1-score"])