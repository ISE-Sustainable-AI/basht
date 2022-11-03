from basht.workload.task_components import Loader, Preprocessor, Splitter, Batcher
from basht.decorators import latency_decorator


class TorchTask:

    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.loader = None
        self.preprocessor_list = []
        self.splitter_list = None
        self.batcher = None
        # TODO: add task default config somewhere

    def add_loader(self, loader: Loader):
        self.loader = loader

    def add_preprocessor(self, preprocessor: Preprocessor):
        self.preprocessor_list.append(preprocessor)

    def add_splitter(self, splitter: Splitter):
        self.splitter = splitter

    def add_batcher(self, batcher: Batcher):
        self.batcher = batcher

    @latency_decorator
    def prepare(self):
        obj_dataset = self.loader.work()
        for preprocessor in self.preprocessor_list:
            obj_dataset = preprocessor.work(obj_dataset)
            self.input_size = obj_dataset.input_size
            self.output_size = obj_dataset.output_size
        for splitter in self.splitter_list:
            obj_dataset = splitter.work(obj_dataset)
        self.train_loader, self.val_loader, self.test_loader = self.batcher.work(obj_dataset)
