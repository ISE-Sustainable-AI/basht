
from time import sleep
from experiments.optuna_minikube.optuna_trial import main
from basht.metrics_storage import MetricsStorage
from basht.resources import Resources


def test_check_trail():
    metrics_storage = MetricsStorage()
    try:
        metrics_storage.start_db()
        sleep(5)

        resource_def = Resources()
        resource_def.trials = 2
        resource_def.workload.epochs = 2

        f = main(resource_def)
        lats = metrics_storage.get_latency_results()
        assert f
        assert len(lats) >= int(resource_def.trials) * 2  # TODO: this is to implicit
    finally:
        metrics_storage.stop_db()

# TODO: do the same for the container ....
# def test_trail_container():
