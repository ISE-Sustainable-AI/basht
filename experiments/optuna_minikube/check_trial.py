
import os
from time import sleep
from experiments.optuna_minikube.optuna_trial import main
from ml_benchmark.metrics_storage import MetricsStorage


def check_trail():
    metrics_storage = MetricsStorage()
    try:
        metrics_storage.start_db()
        sleep(5)
        os.environ["METRICS_STORAGE_HOST"] = MetricsStorage.host
        os.environ["DB_CONN"] = MetricsStorage.connection_string
        os.environ["N_TRIALS"] = "10"
        os.environ["EPOCHS"] = "1"

        f = main()
        assert f

        lats = metrics_storage.get_latency_results()
        assert len(lats) >= int(os.environ["N_TRIALS"])*2 #(validate+train)
    finally:
        metrics_storage.stop_db()

#TODO: do the same for the container ....
# def test_trail_container():
