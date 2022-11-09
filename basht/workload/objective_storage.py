from dataclasses import dataclass


@dataclass
class ObjectiveStorage:

    current_epoch: int = None
    epoch_losses: list = []
    validation_scores: dict = {}


class ObjectiveStorageInterface:

    def __init__(self, objective) -> None:
        self.objective_storage = objective.objective_storage

    def get_current_epoch(self) -> int:
        return self.objective_storage.current_epoch

    def get_epoch_losses(self) -> list:
        return self.objective_storage.epoch_losses

    def get_validation_scores(self):
        return self.objective_storage.validation_scores
