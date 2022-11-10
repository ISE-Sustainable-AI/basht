
import os
from time import sleep
from experiments.optuna_minikube.optuna_trial import main
from basht.metrics_storage import MetricsStorage
from basht.utils.yaml import YMLHandler


def test_check_trail():
    metrics_storage = MetricsStorage()
    try:
        metrics_storage.start_db()
        sleep(5)
        os.environ["METRICS_STORAGE_HOST"] = MetricsStorage.host
        os.environ["DB_CONN"] = MetricsStorage.connection_string


        # resource_path = os.path.join(os.path.dirname(__file__), "resource_definition.yml")
        # resource_def = YMLHandler.load_yaml(resource_path)
        resource_def = {
            "trials": 2,
            "workload": {
                "dl_framework": "torch",
                "task": {
                    "loader": "mnist",
                    "preprocessors": ["ImageFlattner"],
                    "splitter": {
                        "type": "StandardSplitter",
                        "config": {
                            "val_split": 0.2,
                            "test_split": 0.2,
                        },
                    },
                    "batcher": {
                        "type": "StandardBatcher",
                        "config": {
                            "train_batch_size": 50,
                            "val_batch_size": 50,
                            "test_batch_size": 50,
                        },
                    },
                },
                "model_cls": "mlp",
                "epochs": 2,
                "device": "cpu",

            },
            "hyperparameter": {
                "learning_rate": {
                    "start": 1e-4,
                    "step_size": 1e-3,
                    "end": 1e-2
                },

            }
        }

        f = main(dict(resource_def))
        assert f

        lats = metrics_storage.get_latency_results()
        assert len(lats) >= int(resource_def.get("trials")) * \
            2  # (validate+train)
    finally:
        metrics_storage.stop_db()

# TODO: do the same for the container ....
# def test_trail_container():
