from dataclasses import dataclass, field


@dataclass
class ObjectiveStorage:

    current_epoch: int = None
    validation_scores: dict = field(default_factory=list)
    training_scores: dict = field(default_factory=list)

    def add_validation_scores(self, value: dict):
        if isinstance(value, dict):
            self.validation_scores.append(value)
        else:
            raise AttributeError("Validation Scores needs to be a dict.")

    def add_training_scores(self, value: float):
        if isinstance(value, float):
            self.training_scores.append(value)
        else:
            raise AttributeError("Training Scores needs to be a dict.")

    def get_current_epoch_results(self) -> tuple:
        if self.current_epoch < len(self.training_scores):
            training_results = self.training_scores[self.current_epoch]
        if self.current_epoch < len(self.validation_scores):
            validation_results = self.validation_scores[self.current_epoch]
        else:
            validation_results = None
        return training_results, validation_results


class ObjectiveStorageInterface:
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
