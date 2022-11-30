import numpy as np
import itertools
from basht.config import Path
import os


def generate_hidden_layer_config_space(hidden_layer_dict):
    start = hidden_layer_dict.get("start")
    end = hidden_layer_dict.get("end")
    step_size = hidden_layer_dict.get("step_size")
    neuron_steps = step_size[0]

    if neuron_steps or neuron_steps > 0:
        neuron_count_span = range(min(start), max(end)+1, neuron_steps)
    else:
        neuron_count_span = start
    layer_steps = step_size[1]
    combinations = []

    for layer_number in range(len(start), len(end)+1, layer_steps):
        combination = itertools.product(neuron_count_span, repeat=layer_number)
        combinations.extend(list(combination))
    combinations = {i: combinations[i] for i in range(len(combinations))}
    return combinations


def generate_grid_search_space(hyperparameter: dict) -> dict:
    modified_search_space = {}
    for key, value in hyperparameter.items():
        if key == "hidden_layer_config":
            combinations = generate_hidden_layer_config_space(value)
            modified_search_space[key] = combinations
        else:
            item_search_space = np.arange(value["start"], value["end"], value["step_size"]).tolist()
            modified_search_space[key] = np.append(item_search_space, value["end"]).tolist()
    return modified_search_space


if __name__ == "__main__":
    search_space = {
        "learning_rate": {
            "start": 1e-2,
            "end": 5e-2,
            "step_size": 1e-2
        },
        "hidden_layer_config": {
            "start": [10],
            "end": [10, 30],
            "step_size": [10, 1]
        }
    }

    search_space = generate_grid_search_space(search_space)
    print(search_space)
