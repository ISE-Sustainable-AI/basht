from string import Template
import yaml


class YamlTemplateFiller:

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_and_fill_yaml_template(yaml_path: str, yaml_values: dict) -> dict:
        """Loads a YAML-Template File with placeholders in it and returns and object with filled placeholder
        values. Values are gathered from a provided dictionary.

        Args:
            yaml_path (_type_): _description_
            yaml_values (_type_): _description_

        Returns:
            _type_: _description_
        """
        with open(yaml_path, "r") as f:
            job_template = Template(f.read())
        return yaml.safe_load_all(job_template.substitute(yaml_values))

    @staticmethod
    def as_yaml(yaml_path: str,obj : object) -> None:
        """Safely writes an object to a YAML-File.
        Args:
            yaml_path (str): filename to write yaml to
            obj (any): object to save as yaml
        """
        with open(yaml_path, "w") as f:
            f.write("# generated file - do not edit\n")
            yaml.dump(obj, f)
        