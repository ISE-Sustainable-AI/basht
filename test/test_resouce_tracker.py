
import logging
from basht.resource_tracker import ResourceTracker
from basht.metrics_storage import LoggingStoreStrategy
import pytest


def test_resouce_tracker(prometeus_url):
    import time
    if not prometeus_url:
        pytest.skip("Prometheus URL is None on this local machine. Test skipped.")
    logging.basicConfig(level=logging.DEBUG)
    rt = ResourceTracker(prometheus_url=prometeus_url, resouce_store=LoggingStoreStrategy)
    rt.start()
    time.sleep(ResourceTracker.UPDATE_INTERVAL * 15)
    rt.stop()
    print(rt.store.log)
    assert rt.store.log != []


def test_resouce_tracker_with_namespace(prometeus_url):
    import time
    if not prometeus_url:
        pytest.skip("Prometheus URL is None on this local machine. Test skipped.")
    logging.basicConfig(level=logging.DEBUG)
    rt = ResourceTracker(prometheus_url=prometeus_url, resouce_store=LoggingStoreStrategy)
    rt.namespace = "optuna-study"
    rt.start()
    time.sleep(ResourceTracker.UPDATE_INTERVAL * 15)
    rt.stop()
    print(rt.store.log)
    assert rt.store.log != []
