from ml_benchmark.workload.torch_objective import Objective


class ObjectiveDirector:

    def __init(self) -> None:
        self._builder = None

    def build_objective(self) -> None:
        self._builder.build_task()
        self._builder.build_model()

    def get_objective(self) -> Objective:
        return self._builder.objective

    def define_objective_building(self):
        pass
