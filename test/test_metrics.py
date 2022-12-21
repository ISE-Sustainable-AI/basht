import logging
from basht.metrics_storage import MetricsStorage
from basht.resource_tracker import ResourceTracker
from time import sleep


def test_metrics(prometeus_url, prepared_objective):
    # setup
    objective = prepared_objective
    metrics_storage = MetricsStorage()
    resourceTracker = ResourceTracker(prometheus_url=prometeus_url)
    try:
        metrics_storage.start_db()
        sleep(5)
        resourceTracker.start()
        objective.load()
        objective.train()
        vali_score = objective.validate()
        test_score = objective.test()

        sleep(15)

        result = metrics_storage.get_benchmark_results()
        logging.info(result)

        assert len(result["latency"]) > 0
        assert len(result["classification"]) == 2
        assert len([entry for entry in result["classification"] if entry["function_name"] == "test"]) == 1
        if prometeus_url:
            assert len(result["resources"]) > 0
        assert isinstance(vali_score, dict)
        assert isinstance(test_score, dict)
    except Exception as e:
        assert False, e
    finally:
        resourceTracker.stop()
        metrics_storage.stop_db()
