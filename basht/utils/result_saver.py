import shutil
import os
import json


class ResultSaver:

    def __init__(self, experiment_path=None, experiment_name="") -> None:
        if experiment_path:
            self.experiment_path = self._create_experiment_folder(experiment_path, experiment_name)
        else:
            raise AttributeError("No Save Path Supplied!")

    def _create_experiment_folder(
            self, experiment_path, exp_name, overwrite=True):
        folder_name = f"exp__{exp_name}"
        experiment_path = os.path.join(experiment_path, folder_name)
        if os.path.exists(experiment_path) and overwrite:
            shutil.rmtree(experiment_path)
        os.mkdir(experiment_path)
        return experiment_path

    def save_results(self, results):
        with open(os.path.join(self.experiment_path, "benchmark_results.json"), "w") as f:
            json.dump(results, f)
