from yaml import load, Loader


class YMLParser:

    @staticmethod
    def parse(hyperparameter_file_path):
        with open(hyperparameter_file_path, "r") as f:
            hyper_dict = load(f, Loader=Loader)
        return hyper_dict


if __name__ == "__main__":
    from ml_benchmark.config import Path
    import os
    yml_dict = YMLParser.parse(os.path.join(Path.root_path, "example/grid_search/hyper_params.yml"))
