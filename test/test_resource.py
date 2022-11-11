from os import path
from basht.utils.yaml import YMLHandler
from basht.resources import Resouces

def test_resource():
    res = Resouces()

    assert res.trials == 100
    assert res.workload.device == "cpu"



def test_yaml():
    resources_path = path.join(path.dirname(__file__), "test.yaml")
    resources = Resouces.from_yaml(resources_path)

    #check read value
    assert resources.trials == 5
    #check default collision (same value as in default)
    assert resources.workload.device == "cpu"

    #check default value for missing key
    assert resources.workload.task.loader == "mnist"

    #check extra key
    assert resources.args["kubernetesContext"]