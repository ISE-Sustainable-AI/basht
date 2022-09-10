import logging
from ml_benchmark.latency_tracker import Tracker #TODO: move to utils
from ml_benchmark.metrics import Result
from ml_benchmark.metrics_storage import MetricsStorageStrategy

class ResultTracker(Tracker):
    def __init__(self,store=MetricsStorageStrategy):
        self.store = store()
        self.store.setup()

    def track(self, objective_function, result):
        r = Result(objective=objective_function)
    
        r.value = result["macro avg"]["f1-score"]
        r.measure = "f1-score"

        r.hyperparameters = objective_function.__self__.get_hyperparameters()
        r.classification_metrics = result

        try:
            self.store.store(r,table_name="classification_metrics")
            logging.info("Stored result")
        except Exception as e:
            logging.error(f"failed to store result {e}")
