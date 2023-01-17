from basht.decorators import latency_decorator
from basht.metrics_storage import MetricsStorage


def test_exit_exception(exception_function):
    # setup
    value = True
    metrics_storage = MetricsStorage()

    @latency_decorator
    def decorated_function(func, value):
        result = func(value)
        return result

    # perform
    metrics_storage.start_db()
    try:
        decorated_function(exception_function, value)
    except AttributeError:
        pass
    result = metrics_storage.get_latency_results()
    metrics_storage.stop_db()
    # perform

    assert result
    assert result[0]["aborted"]
