import logging
from basht.latency_tracker import Tracker #TODO: move to utils
from basht.metrics import Result
from basht.metrics_storage import MetricsStorageStrategy


class ResultTracker(Tracker):
    def __init__(self,store=MetricsStorageStrategy):
        self.store = store()
        self.store.setup()

    def track(self, objective_function, result):
        r = Result(objective=objective_function)

        r.value = result["macro avg"]["f1-score"]
        r.measure = "f1-score"

        r.hyperparameters = objective_function.__self__.hyperparameter  # function needs to be part of an object
        r.classification_metrics = result

        try:
            self.store.store(r,table_name="classification_metrics")
            logging.info("Stored result")
        except Exception as e:
            logging.error(f"failed to store result {e}")
