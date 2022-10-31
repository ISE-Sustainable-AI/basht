class TorchTask:

    def __init__(self):
        self.seed = 1337
        self.component_list = list()

    def add_component(self, component):
        self.component_list.append(component)

    def _set_input_output_format(self):
        pass

    def prepare_task(self):
        component_output = self.component_list.pop(0)()
        for component in self.component_list:
            component_output = component.work(component_output)
