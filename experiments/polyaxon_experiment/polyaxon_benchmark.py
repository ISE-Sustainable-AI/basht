from ast import dump
import json
from os import path
import os
import sys
from textwrap import indent
from time import sleep
from urllib import request
from kubernetes.client.rest import ApiException
import random
from kubernetes import client, config
from string import Template
import yaml
import docker
PROJECT_ROOT = path.abspath(path.join(__file__ ,"../../.."))
sys.path.append(PROJECT_ROOT)
from ml_benchmark.benchmark_runner import Benchmark
import requests




class PolyaxonBenchmark(Benchmark):

    def __init__(self, objective, grid, resources) -> None:
        self.objective = objective
        self.grid = grid
        self.resources = resources
        self.group="kubeflow.org"
        self.version="v1beta1"
        self.namespace='kubeflow'
        self.plural="experiments"
        self.experiment_file_name = "grid.yaml"
        self.project_description = "Na chuj mnie te piura"
        self.polyaxon_addr="http://localhost:31833/"

        config.load_kube_config()

        

    
        if "dockerImageTag" in self.resources:
            self.trial_tag = self.resources["dockerImageTag"]
        else:
            self.trial_tag = "mnist_polyaxon:latest"

        if "dockerUserLogin" in self.resources:
            self.docker_user = self.resources["dockerUserLogin"]
            self.trial_tag =  f'{self.docker_user}/{self.trial_tag}'
        else:
            self.docker_user = ""
            self.trial_tag = "witja46/mnist_polyaxon:latest"
        
        if "dockerUserPassword" in self.resources:
            self.docker_pasword = self.resources["dockerUserPassword"]
        else:
            self.docker_pasword = ""        


        if "studyName" in self.resources:
            self.study_name = self.resources["studyName"]
        else:
            self.study_name = f"polyaxon-study-{random.randint(0, 100)}"

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

        if "jobsCount" in resources:
            self.jobsCount = self.resources["jobsCount"]
        else:
            self.jobsCount = 6
        
    def deploy(self):
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """
        
        print("Adding polyaxon to helm repo:")
        res = os.popen('helm repo add polyaxon https://charts.polyaxon.com').read()
        print(res)

        #TODO deploy via helm for better error handling and easier configuration
        print("Deploying polyaxon to minikube:")
        res = os.popen('polyaxon admin deploy -t minikube').read()
        print(res)

        print("Checking deployment status")
        res = os.popen('kubectl get deployment -n polyaxon').read()
        print(res)
      


        config.load_kube_config()
        api = client.CoreV1Api()
        
        # waiting while not all polyaxon pods are runing 
        all_running = False
        while not all_running:
            sleep(1)
            resource = api.list_namespaced_pod(namespace="polyaxon")
            all_running = True
            for pod in resource.items:
                print(f'{pod.metadata.name}  {pod.status.phase}')
                if (pod.status.phase != "Running"):
                    all_running = False

        #TODO start subprocess for polyaxon port-forward -t minikube
        # res = os.popen('polyaxon port-forward -t minikube').read()
        # print(res)




      
        
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

        #loading and filling the template
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

        #TODO add error handling 
        print("Creating new project:")
        project = requests.post(f'{self.polyaxon_addr}/api/v1/default/projects/create', json={"name": self.study_name, "description": self.project_description})
        print(project.text)


        #TODO find out where to pass the --eager flag so that the run can be created with help of api
        # with open(path.join(path.dirname(__file__), self.experiment_file_name), "r") as f:
        #     body = f.read()
        #     print(body)
        #     run = requests.post(f'{self.polyaxon_addr}/api/v1/default/{self.study_name}/runs',json={"content":body,"eager":True ,"tags":["--eager"],"meta_info":{"eager":True,"--eager":True,"flags":"--eager"}})
        #     print(run.text)
      
      
        print("Starting polyaxon experiment:")
        res = os.popen(f'polyaxon run -f ./{self.experiment_file_name} --project {self.study_name} --eager').read()
        print(res)

        

        print("Waiting for the run to finish")
        finished = False
        while not finished:
            runs = self.get_succeeded_runs()
            print(f'{runs["count"]} jobs out of {self.jobsCount} succeded')
            
            #checking if all runs were finished
            finished = runs["count"] == self.jobsCount
            sleep(1)
        return 


    def collect_benchmark_metrics(self):
        pass



    def get_succeeded_runs(self, sort_by="duration"):
        
        #TODO add error handling acording to polyaxon api 
        res = requests.get(f'{self.polyaxon_addr}/api/v1/default/{self.study_name}/runs?query=status:succeeded&sort={sort_by}') 
        result = json.loads(res.text)
        return result

    def collect_run_results(self):
        

        print("Collecting run results:")
        result = self.get_succeeded_runs()
        print(json.dumps(result,indent=4))               
               
        print("\n Experiment finished with following optimal trial:")
        print(result["results"][0])
        return result["results"][0]
    
    def test(self):
        return super().test()

    def undeploy(self):
        print("Undeploying polyaxon")
        #TODO run comand polyaxon admin teardown plus enter y

        #TODO wait untill all pods get terminated



      



if __name__ == "__main__":
    #main()
    bench = PolyaxonBenchmark(1,1,resources={
        #  "dockerUserLogin":"witja46",
        #  "dockerUserPassword":"J$rmakowicz1998",
        # "studyName":"jprd1"
        "jobsCount":22,
        "workerCount":20
        })
 #   bench.deploy()
    bench.setup()
    bench.run()

    bench.collect_run_results()

   # bench.setup()
    #bench.run()
      #  deploy()   
      #    try:
    #bench.collect_run_results() config.load_kube_config() group="kubeflow.org", version="v1beta1", namespace="kubeflow",
        # get the cluster scoped resource
    # bench.collect_run_results()
    # print(resource) 
  
    # all_running = False
    # while not all_running:

    #     config.load_kube_config()
    #     api = client.CoreV1Api()
    #     resource = api.list_namespaced_pod(namespace="polyaxon")
    #     all_running = True
    #     for pod in resource.items:
    #         print(f'{pod.metadata.name}  {pod.status.phase}')
    #         if (pod.status.phase != "Running"):
    #             all_running = False
# print(resource["status"]["currentOptimalTrial"
    # bench.run()
    # api = client.CustomObjectsApi()

    # it's my custom resource defined as Dict
    # my_resource = {
    #     "apiVersion": "stable.example.com/v1",
    #     "kind": "CronTab",
    #     "metadata": {"name": "my-new-cron-object2"},
    #     "spec": {
    #         "cronSpec": "* * * * */5",
    #         "image": "my-awesome-cron-image"
    #     }
    # }

    # patch to update the `spec.cronSpec` field
    # patch_body = {
    #     "spec": {"cronSpec": "* * * * */10", "image": "my-awesome-cron-image"}
    # }

    # create the resource
    # try:
    #     res = api.create_namespaced_custom_object(
    #         group="stable.example.com",
    #         version="v1",
    #         namespace="default",
    #         plural="crontabs",
    #         body=my_resource,
    #     )
    #     print(res)
    #     print("Resource created")
    # except ApiException as e:
    #     print("Exception when calling CustomObjectsApi->create_cluster_custom_object: %s\n" % e)
    
    # bench.run()
    


     

