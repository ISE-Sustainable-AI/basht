
from basht.latency_tracker import LatencyTracker
from basht.metrics import Latency
from basht.results_tracker import ResultTracker


def validation_latency_decorator(func):
    """
    A Decorator to record the latency of the decorated function. Once it is recorded the LatencyTracker
    writes the result into the postgres database.

    We assume that that the decorated function returns a dictionary with the following keys:
        - "macro avg": the macro average of the validation with the keys:
            - "f1-score": the f1-score

    """
    def result_func(*args, **kwargs):
        func.__self__ = args[0]
        with Latency(func) as latency:
            result = func(*args, **kwargs)
        latency_tracker = LatencyTracker()
        tracker = ResultTracker()

        latency_tracker.track(latency)
        # XXX this locks us into the f1-score, we probably want to track all callification metrics not just f1-score. MG please help :)
        tracker.track(func, result)
        func.__self__ = None
        return result

    return result_func


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
