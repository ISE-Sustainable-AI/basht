from os import path
from basht.utils.yaml import YMLHandler, YamlTemplateFiller


def test_yaml():
    resources = YMLHandler.load_yaml(path.join(path.dirname(__file__), "test.yaml"))
    assert resources["deleteAfterRun"]

    YMLHandler.as_yaml(
        path.join(path.dirname(__file__), "hyperparameter_space.yml"), resources["hyperparameter"])
    params = YMLHandler.load_yaml(path.join(path.dirname(__file__), "hyperparameter_space.yml"))
    assert params == resources["hyperparameter"]


def test_yaml_template_filler():
    template_filler = YamlTemplateFiller()
    yml_path = path.join(path.dirname(__file__), "test.yaml")
    fill_value = dict(pruning=None)

    filled_template = template_filler.load_and_fill_yaml_template(yml_path, fill_value, as_dict=True)

    assert filled_template["pruning"] is None
