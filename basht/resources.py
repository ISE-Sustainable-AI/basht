from dataclasses import asdict, dataclass, field
from typing import List, Literal, Union
import yaml

@dataclass
class SplitterConfig:
    val_split: float = 0.2
    test_split: float = 0.2

@dataclass
class Splitter:
    type: str = "StandardSplitter"
    config: SplitterConfig = SplitterConfig()

@dataclass
class BatcherConfig:
    train_batch_size: int = 50
    val_batch_size: int = 50
    test_batch_size: int = 50

@dataclass
class Batcher:
    type: str = "StandardBatcher"
    config: BatcherConfig = BatcherConfig()



@dataclass
class Task:
    preprocessors: List = field(default_factory=lambda: ["ImageFlattner"])
    loader: str = "mnist"
    
    splitter: 'Splitter' = Splitter()
    batcher: 'Batcher' = Batcher()
    


@dataclass
class HiddenLayerConfig:
    start: Union[float, List[int]]
    end: Union[float, List[int]]
    step_size: Union[float, List[int]]

@dataclass
class Hyperparameter:
    learning_rate: 'HiddenLayerConfig' = HiddenLayerConfig(start=1e-4, end=1e-2, step_size=1e-3)
    weight_decay: 'HiddenLayerConfig' = HiddenLayerConfig(start=1e-6, end=1e-4, step_size=1e-5)
    hidden_layer_config: 'HiddenLayerConfig' = HiddenLayerConfig(start=[10], end=[100, 100,100], step_size=[10, 1])

@dataclass
class Workload:
    epochs: int = 100
    task: Task = Task()
    dl_framework: str = "torch"
    model_cls: str = "mlp"
    device: str = "cpu"

@dataclass
class Resouces:
    """
        Resource definition for a HPO benchmark

        Args:
            trials (int): Number of trials to run - up to the HPO 
            framework to enforce
           
            metrics_ip (str): IP address of the metrics server (usually the same as the benchmark runner)
            
            worker_cpu (int): Number of CPUs to allocate to each worker
            worker_memory (int): Amount of memory to allocate to each worker in GB
            worker_count (int): Number of workers to spawn (up to the HPO platform to translate to nodes)

            workload (Workload): Workload definition for the benchmark, including the task, model, preprocessing, etc.

            hyperparameter (Hyperparameter): Hyperparameter definition for the benchmark, including the search space size, etc.
    """
    metrics_ip: str = "auto" #TODO we should instead use a factory here

    trials: int = 100

    worker_cpu: int = 2
    worker_memory: int = 2
    worker_count: int = 1

    
    workload: 'Workload' = Workload()
    hyperparameter: 'Hyperparameter' = Hyperparameter()

    def to_dict(self):
        return asdict(self)
    
    def to_yaml(self):
        return yaml.dump(self.to_dict())



