import logging
from basht.metrics_storage import MetricsStorage
from basht.resource_tracker import ResourceTracker
from basht.results_tracker import ResultTracker
from time import sleep


def test_metrics(prometeus_url, prepared_objective):
    # setup
    objective = prepared_objective
    metrics_storage = MetricsStorage()
    resourceTracker = ResourceTracker(prometheus_url=prometeus_url)
    result_tracker = ResultTracker()
    try:
        metrics_storage.start_db()
        sleep(5)
        resourceTracker.start()
        objective.load()
        objective.train()
        score = objective.validate()
        result = objective.test()
        result_tracker.track(objective.test, result)

        sleep(15)

        result = metrics_storage.get_benchmark_results()
        logging.info(result)

        assert len(result["latency"]) > 0
        assert len(result["classification"]) == 2
        assert len([entry for entry in result["classification"] if entry["function_name"] == "test"]) > 0
        if prometeus_url:
            assert len(result["resources"]) > 0
        assert isinstance(score, dict)
    except Exception as e:
        assert False, e
    finally:
        resourceTracker.stop()
        metrics_storage.stop_db()
