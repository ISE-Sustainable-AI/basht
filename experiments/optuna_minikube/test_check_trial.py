
import os
from time import sleep
from experiments.optuna_minikube.optuna_trial import main
from basht.metrics_storage import MetricsStorage
from basht.utils.yaml import YMLHandler
from basht.resources import Resouces

def test_check_trail():
    metrics_storage = MetricsStorage()
    try:
        metrics_storage.start_db()
        sleep(5)
        os.environ["METRICS_STORAGE_HOST"] = MetricsStorage.host
        os.environ["DB_CONN"] = MetricsStorage.connection_string

        resource_def = Resouces()
        resource_def.trials = 2
        resource_def.workload.epochs = 2

        f = main(resource_def.to_dict())
        assert f

        lats = metrics_storage.get_latency_results()
        assert len(lats) >= int(resource_def.trials) * \
            2  # (validate+train)
    finally:
        metrics_storage.stop_db()

# TODO: do the same for the container ....
# def test_trail_container():
