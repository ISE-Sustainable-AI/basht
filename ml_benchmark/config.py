import os
from pathlib import Path
from dataclasses import dataclass, asdict, field


class Path:

    file_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = Path(file_dir).parent
    data_path = os.path.join(root_path, "data")
    experiments_path = os.path.join(root_path, "experiments")


@dataclass
class MnistConfig:
    val_split_ratio: float = 0.2
    train_batch_size: int = 512
    val_batch_size: int = 128
    test_batch_size: int = 128
    epochs: int = 100

    def to_dict(self):
        return asdict(self)


@dataclass
class MLPHyperparameter:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    hidden_layer_config: list = field(default_factory=lambda: [15])

    def to_dict(self):
        return asdict(self)


class MetricsStorageConfig:
    port = 5432
    user = "root"
    password = "1234"
    db = "benchmark_metrics"
    host = "localhost"
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
