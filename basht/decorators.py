
from basht.latency_tracker import LatencyTracker
from basht.metrics import Latency
from basht.results_tracker import ResultTracker


def latency_decorator(func):
    """A Decorator to record the latency of the decorated function. Once it is recorded the LatencyTracker
    writes the result into the postgres databse.

    Decorators overwrite a decorated function once the code is passed to the compier

    Args:
        func (_type_): _description_
    """
    def latency_func(*args, **kwargs):
        result = None
        func.__self__ = args[0]
        latency_tracker = LatencyTracker()
        try:
            with Latency(func) as latency:
                result = func(*args, **kwargs)
        except Exception as e:
            latency.stop()
            latency._calculate_duration()
            latency._mark_as_aborted()
            raise e
        finally:
            latency_tracker.track(latency)
            func.__self__ = None
        return result
    return latency_func


def result_tracking_decorator(func):
    """This function tracks classification results of a given function. It intended to be used for validation
    or test functions of objectives. An numpy classification metric dict should be returned by these functions

    Args:
        func (_type_): _description_
    """
    def result_func(*args, **kwargs):
        func.__self__ = args[0]
        result_tracker = ResultTracker()
        result = func(*args, **kwargs)
        result_tracker.track(func, result)
        func.__self__ = None
        return result

    return result_func
