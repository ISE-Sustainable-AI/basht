import logging
from os import path
from ml_benchmark.utils.yml_parser import YMLParser
from ml_benchmark.utils.yaml_template_filler import YamlTemplateFiller

def test():
     resources = YMLParser.parse(path.join(path.dirname(__file__),"test.yaml"))
     assert resources["deleteAfterRun"]

     print(resources["hyperparameter"])

     YamlTemplateFiller.as_yaml(path.join(path.dirname(__file__),"hyperparameter_space.yml"), resources["hyperparameter"])
     params = YMLParser.parse(path.join(path.dirname(__file__),"hyperparameter_space.yml"))
     assert params == resources["hyperparameter"]

     