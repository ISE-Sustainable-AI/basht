from basht.workload.task_components.loader import Loader


class TorchTask:

    def __init__(self):
        self.seed = 1337
        self.component_list = list()

    def add_component(self, component):
        self.component_list.append(component)

    def _set_input_output_format(self):
        pass

    def prepare_task(self):
        loader = self.component_list.pop(0)
        if isinstance(loader, Loader):
            component_output = loader.work()
        for component in self.component_list:
            component_output = component.work(component_output)
        self.loader_tuple = component_output
