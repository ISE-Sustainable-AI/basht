from basht.config import WorkloadEnums


class WorkloadValidator:

    @staticmethod
    def validate(self, workload_definition: dict) -> dict:
        return {key: WorkloadEnums(value) for key, value in workload_definition.items()}
