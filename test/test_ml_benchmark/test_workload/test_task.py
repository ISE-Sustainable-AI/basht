from symbol import testlist_comp
from basht.workload.task import TorchTask
from basht.workload.task_components.loader import MnistLoader
from basht.workload.task_components.preprocessor import TorchImageFlattner
from basht.workload.task_components.splitter import StandardTorchSplitter
from basht.workload.task_components.batcher import LoaderTuple, TorchBatcher


def test_prepare_task():
    # setup
    task = TorchTask()
    task.add_component(MnistLoader())
    task.add_component(TorchImageFlattner())
    task.add_component(StandardTorchSplitter(vali_split=0.2, test_split=0.2))
    task.add_component(TorchBatcher(train_batch_size=10, vali_batch_size=10,
                                    test_batch_size=10))

    # work
    task.prepare_task()

    # check
    assert isinstance(task.loader_tuple, LoaderTuple)
    assert task.loader_tuple.train_loader
    assert task.loader_tuple.vali_loader
    assert task.loader_tuple.test_loader
