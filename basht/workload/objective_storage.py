from dataclasses import dataclass, field


@dataclass
class ObjectiveStorage:

    current_epoch: int = None
    validation_scores: dict = field(default_factory={})
    training_scores: dict = field(default_factory={})

    def add_validation_scores(self, value):
        if isinstance(value, dict):
            self.validation_scores.update(value)
        else:
            raise AttributeError("Validation Scores needs to be a dict.")

    def add_training_scores(self, value):
        if isinstance(value, dict):
            self.training_scores.update(value)
        else:
            raise AttributeError("Training Scores needs to be a dict.")


class ObjectiveStorageInterface:
    # TODO: avoid self reference, objective is gonna be in objective if executed in objective - check again
    def __init__(self, objective) -> None:
        self.objective_storage = objective.objective_storage

    def get_current_epoch(self) -> int:
        return self.objective_storage.current_epoch

    def get_training_scores(self) -> dict:
        return self.objective_storage.training_scores

    def get_validation_scores(self) -> dict:
        return self.objective_storage.validation_scores
