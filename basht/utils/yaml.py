from string import Template
import ruamel.yaml
from ruamel.yaml import YAML


class YMLHandler:

    @staticmethod
    def load_yaml(file_path):
        """Safely writes an object to a YAML-File.
        Args:
            yaml_path (str): filename to write yaml to
            obj (any): object to save as yaml
        """
        with open(file_path, "r") as f:
            file_dict = ruamel.yaml.safe_load(f)
        return file_dict

    @staticmethod
    def as_yaml(yaml_path: str, obj: object) -> None:
        """Safely writes an object to a YAML-File.
        Args:
            yaml_path (str): filename to write yaml to
            obj (any): object to save as yaml
        """
        with open(yaml_path, "w") as f:
            f.write("# generated file - do not edit\n")
            ruamel.yaml.dump(obj, f, Dumper=ruamel.yaml.RoundTripDumper)


class NullRepresenter:
    def __init__(self):
        self.count = 0

    def __call__(self, repr, data):
        ret_val = repr.represent_scalar(u'tag:yaml.org,2002:null', u'null' if self.count == 0 else u'')
        self.count += 1
        return ret_val


class YamlTemplateFiller:

    def __init__(self) -> None:
        self.yaml_handler = YMLHandler()

    @staticmethod
    def load_and_fill_yaml_template(yaml_path: str, yaml_values: dict, as_dict: bool = False) -> dict:
        """Loads a YAML-Template File with placeholders in it and returns and object with filled placeholder
        values. Values are gathered from a provided dictionary.

        Args:
            yaml_path (_type_): _description_
            yaml_values (_type_): _description_

        Returns:
            _type_: _description_
        """

        yaml_values = YamlTemplateFiller._adjust_substitute_for_yml(yaml_values=yaml_values)
        with open(yaml_path, "r") as f:
            job_template = Template(f.read())
        yaml = YAML()
        null_representer = NullRepresenter()
        yaml.representer.add_representer(type(None), null_representer)
        filled_template = yaml.load_all(job_template.substitute(yaml_values))
        if as_dict:
            filled_template = next(filled_template)
        return filled_template

    @staticmethod
    def _adjust_substitute_for_yml(yaml_values):
        for key, value in yaml_values.items():
            if not value:
                yaml_values[key] = "null"
        return yaml_values

