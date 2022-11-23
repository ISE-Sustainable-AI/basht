import os
from pathlib import Path


class Path:

    file_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = Path(file_dir).parent
    data_path = os.path.join(root_path, "data")
    experiments_path = os.path.join(root_path, "experiments")
    test_path = os.path.join(root_path, "test")


class MetricsStorageConfig:
    port = 5432
    user = "root"
    password = "1234"
    db = "benchmark_metrics"
    host = "localhost"
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{db}"
