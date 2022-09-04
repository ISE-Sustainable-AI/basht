import numpy as np
from ml_benchmark.utils.yml_parser import YMLParser
import itertools

def generate_search_space(yaml_file_path):
    search_space = YMLParser.parse(yaml_file_path)
    modified_search_space = {}
    hidden_layer_config = []
    for key, value in search_space.items():
        if isinstance(value["start"], list):
            combinations = []
            numbers = range(value["start"][0], value["end"][-1], value["step_size"][0])
            for r in range(len(value["end"])):
                r = r + 1
                for combination in itertools.combinations(set(numbers), r):
                    combinations.append(list(combination))


            modified_search_space[key] = combinations
        else:
            modified_search_space[key] = np.arange(value["start"], value["end"], value["step_size"])
    return modified_search_space
