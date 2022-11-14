from dataclasses import dataclass, field


@dataclass
class ObjectiveStorage:

    current_epoch: int = None
    validation_scores: dict = field(default_factory=list)
    training_scores: dict = field(default_factory=list)

    def add_validation_scores(self, value):
        if isinstance(value, dict):
            self.validation_scores.append(value)
        else:
            raise AttributeError("Validation Scores needs to be a dict.")

    def add_training_scores(self, value):
        if isinstance(value, dict):
            self.training_scores.append(value)
        else:
            raise AttributeError("Training Scores needs to be a dict.")

    def get_current_epoch_results(self):
        return self.training_scores[self.current_epoch], self.validation_scores[self.current_epoch]


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


if __name__ == "__main__":
    storage = ObjectiveStorage()
