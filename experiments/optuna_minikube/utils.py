import numpy as np
from basht.utils.yaml import YMLHandler
import itertools
from basht.config import Path
import os


def generate_hidden_layer_config_space(hidden_layer_dict):
    start = hidden_layer_dict.get("start")
    end = hidden_layer_dict.get("end")
    step_size = hidden_layer_dict.get("step_size")

    neuron_steps = step_size[0]
    neuron_count_span = range(min(start), max(end), neuron_steps)
    layer_steps = step_size[1]

    combinations = []

    for layer_number in range(len(start)+1, len(end)+1, layer_steps):
        combination = itertools.combinations_with_replacement(neuron_count_span, layer_number)
        combinations.extend(list(combination))
    combinations.append(tuple(end))
    combinations = {i: combinations[i] for i in range(len(combinations))}
    return combinations


def generate_search_space(hyperparameter):
    modified_search_space = {}
    for key, value in hyperparameter.items():
        if key == "hidden_layer_config":
            combinations = generate_hidden_layer_config_space(value)
            modified_search_space[key] = combinations
        else:
            item_search_space = np.arange(value["start"], value["end"], value["step_size"])
            modified_search_space[key] = np.append(item_search_space, value["end"])
    return modified_search_space


if __name__ == "__main__":
    file_path = os.path.join(Path.root_path, "experiments/optuna_minikube/resource_definition.yml")
    search_space = generate_search_space(file_path)
    print(search_space)
