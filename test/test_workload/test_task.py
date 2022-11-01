from basht.workload.task import TorchTask
from basht.workload.task_components.loader import MnistLoader
from basht.workload.task_components.preprocessor import TorchImageFlattner
from basht.workload.task_components.splitter import StandardTorchSplitter
from basht.workload.task_components.batcher import TorchBatcher


def test_prepare_task():
    # setup
    task = TorchTask()
    task.add_loader(MnistLoader())
    task.add_preprocessor(TorchImageFlattner())
    task.add_splitter(StandardTorchSplitter(val_split=0.2, test_split=0.2))
    task.add_batcher(TorchBatcher(train_batch_size=10, val_batch_size=10, test_batch_size=10))

    # work
    task.prepare_task()

    # check
    assert task.train_loader
    assert task.val_loader
    assert task.test_loader
