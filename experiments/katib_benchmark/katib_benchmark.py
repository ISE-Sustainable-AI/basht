from os import path
import os
import sys
from time import sleep
from kubernetes.client.rest import ApiException
import random
from kubernetes import client, config
from string import Template
import yaml
import docker
PROJECT_ROOT = path.abspath(path.join(__file__ ,"../../.."))
sys.path.append(PROJECT_ROOT)
from ml_benchmark.benchmark_runner import Benchmark




class KatibBenchmark(Benchmark):

    def __init__(self, objective, grid, resources) -> None:
        self.objective = objective
        self.grid = grid
        self.resources = resources
        self.group="kubeflow.org"
        self.version="v1beta1"
        self.namespace='kubeflow'
        self.plural="experiments"
        self.experiment_file_name = "grid.yaml"
     
        config.load_kube_config()

        

    
        if "dockerImageTag" in self.resources:
            self.trial_tag = self.resources["dockerImageTag"]
        else:
            self.trial_tag = "mnist_katib:latest"

        if "dockerUserLogin" in self.resources:
            self.docker_user = self.resources["dockerUserLogin"]
            self.trial_tag =  f'{self.docker_user}/{self.trial_tag}'
        else:
            self.docker_user = ""
            self.trial_tag = "witja46/mnist_katib:latest"
        
        if "dockerUserPassword" in self.resources:
            self.docker_pasword = self.resources["dockerUserPassword"]
        else:
            self.docker_pasword = ""        


        if "studyName" in self.resources:
            self.study_name = self.resources["studyName"]
        else:
            self.study_name = f"katib-study-{random.randint(0, 100)}"

        if "workerCpu" in self.resources:
            self.workerCpu = self.resources["workerCpu"]
        else:
            self.workerCpu = 2

        if "workerMemory" in resources:
            self.workerMemory = self.resources["workerMemory"]
        else:
            self.workerMemory = 2

        if "workerCount" in resources:
            self.workerCount = self.resources["workerCount"]
        else:
            self.workerCount = 5

        if "jobs_Count" in resources:
            self.jobsCount = self.resources["jobs_Count"]
        else:
            self.jobsCount = 6
    def deploy(self):
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """
        print("Deploying katib:")
        res = os.popen('kubectl apply -k "github.com/kubeflow/katib.git/manifests/v1beta1/installs/katib-standalone?ref=master').read()
        print(res)


        #TODO waiit untill all pods are running



      
        
    def setup(self):
        """
        Every Operation that is needed before the actual optimization (trial) starts and that is not relevant
        for starting up workers or the necessary architecture.
        """
       
        #creating experiment yaml          
             
        experiment_definition = {
            "worker_num": self.workerCount,
            "jobs_num":self.jobsCount,
            "worker_cpu": self.workerCpu,
            "worker_mem": f"{self.workerMemory}Gi",
            "worker_image": self.trial_tag,
            "study_name": self.study_name,
            "trialParameters":"${trialParameters.learningRate}"
        }

        #loading and fulling the template
        with open(path.join(path.dirname(__file__), "experiment_template.yaml"), "r") as f:
            job_template = Template(f.read())
            job_yml_objects = job_template.substitute(experiment_definition)
            
        #writing the experiment definition into the file        
        with open(path.join(path.dirname(__file__), self.experiment_file_name), "w") as f:
            f.write(job_yml_objects)
        print("Experiment yaml created")
      
      
        # Creating new docker image if credentials were passed
        if "dockerUserLogin" in self.resources:
           
            #creating task docker image  
            print("Creating task docker image")  
            self.client = docker.client.from_env()
            image, logs = self.client.images.build(path="./mnist_task",tag=self.trial_tag)
            print(f"Image: {self.trial_tag}")
            for line in logs  :
                print(line) 
            
            #pushing to repo
            self.client.login(username=self.docker_user, password=self.docker_pasword)
            for line in self.client.images.push(self.trial_tag, stream=True, decode=True):
                print(line) 
        
        

    def run(self):
        
        print("Starting Katib experiment:")
        api_instance=  client.CustomObjectsApi()

        #Loading experiment definition 
        with open(path.join(path.dirname(__file__), self.experiment_file_name), "r") as f:
            body = yaml.safe_load(f)

            #Starting experiment by creating experiment crd with help of kubernetes API
            try:
                api_response = api_instance.create_namespaced_custom_object(
                    group=self.group,
                    version=self.version,
                    namespace=self.namespace,
                    body=body,
                    plural=self.plural,
                )
                print("Succses: Experiment started")
            #    print(api_response)  
            except ApiException as e:
                print("Exception when calling CustomObjectsApi->create_cluster_custom_object: %s\n" % e)
            
 
    def collect_benchmark_metrics(self):
        return super().collect_benchmark_metrics()



    def get_experiment(self):
        config.load_kube_config()
        api = client.CustomObjectsApi()
        #requesting experiments crd that contains data about finisched experiment
        try:
            resource = api.get_namespaced_custom_object_status(
                    group=self.group,
                    version=self.version,
                    namespace=self.namespace,
                    name=self.study_name,
                    plural=self.plural,
                )
       
            # print(resource)
            # print(resource["status"])
        except ApiException as e:
            print("Exception when calling CustomObjectsApi->get_namespaced_custom_object_status: %s\n" % e)
        return resource


    def collect_run_results(self):
        
        print("Collecting run results:")
        
        sleep(5)
        experiment = self.get_experiment()
        while "status" not in experiment:
            print("Waitinng for the status")
            sleep(5)
            experiment = self.get_experiment()
            print(experiment)
        
    
        while experiment["status"]["conditions"][-1]["type"]!="Succeeded":
            experiment = self.get_experiment()
            print("\nWaiting for the experiment to finish:")
            if "trialsSucceeded" in experiment["status"]:
                print(f'{experiment["status"]["trialsSucceeded"]} trials of {self.jobsCount} succeeded')
            print(experiment["status"]["conditions"][-1])
            sleep(2)
           
       
        
        print("\n Experiment finished with following optimal trial:")
        print(experiment["status"]["currentOptimalTrial"])
         
        
    
    def test(self):
        return super().test()

    def undeploy(self):
        return super().undeploy()

    def main():
        print("jeb")
      



if __name__ == "__main__":
    #main()
    bench = KatibBenchmark(1,1,resources={
        # "dockerUserLogin":"",
        # "dockerUserPassword":"",
        # "studyName":""
        "jobs_Count":20,
        "workerCount":20
        })
    # bench.deploy()
    bench.setup()
    bench.run()
    bench.collect_run_results()



