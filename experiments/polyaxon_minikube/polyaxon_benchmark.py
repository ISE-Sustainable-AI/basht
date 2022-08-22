
from __future__ import print_function
from asyncio import subprocess
from base64 import decode
from cmath import pi
from concurrent.futures import process
from itertools import count
import json
from os import path 
import os
from socket import timeout
import sys
from time import sleep
import random
from urllib.request import urlopen
from venv import create
from kubernetes import client, config,watch
from kubernetes.client.rest import ApiException
from string import Template
import docker
from ml_benchmark.benchmark_runner import Benchmark
from ml_benchmark.utils.image_build_wrapper import builder_from_string
import requests
import subprocess
import psutil
import logging as log
from polyaxon.cli.projects import create
from polyaxon.cli.run import run
from polyaxon.cli.admin import deploy,teardown
from click.testing import CliRunner




class PolyaxonBenchmark(Benchmark):

    def __init__(self, resources) -> None:
        # self.objective = objective
        # self.grid = grid
        self.resources = resources
        self.group="kubeflow.org"
        self.version="v1beta1"
        self.namespace='polyaxon'
        self.plural="experiments"
        self.experiment_file_name = "grid.yaml"
        self.project_description = "Somer random description"
        self.polyaxon_addr="http://localhost:31833/"
        self.post_forward_process=False
        self.cli_runner=CliRunner()

        config.load_kube_config()

        self.logging_level= self.resources.get("loggingLevel",log.CRITICAL)

        self.create_clean_image = self.resources.get("createCleanImage",True) 
        log.basicConfig(format='%(asctime)s Polyaxon Benchmark %(levelname)s: %(message)s',level=self.logging_level)


        self.metrics_ip = resources.get("metricsIP")
    
        if "dockerImageTag" in self.resources:
            self.trial_tag = self.resources["dockerImageTag"]
        else:
            self.trial_tag = "mnist_task"

        if "dockerUserLogin" in self.resources:
            self.docker_user = self.resources["dockerUserLogin"]
        else:
            self.docker_user = "witja46"
    
        
        
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
        
        log.info("Adding polyaxon to helm repo:")
        res = os.popen('helm repo add polyaxon https://charts.polyaxon.com').read()
        log.info(res)

        log.info("Deploying polyaxon to minikube:")
        #invoking polyaxon cli deploy comand
        res = self.cli_runner.invoke(deploy)
        log.info(res.output)

      
      
        

        config.load_kube_config()
        w = watch.Watch()
        c = client.CoreV1Api()
        deployed = 0


        log.info("Waiting for all polyaxon pods to be ready:")
        # From all pods that polyaxon starts we are onlly really intrested for following 4 that are crucial for runnig of the experiments 
        monitored_pods = ["polyaxon-polyaxon-streams","polyaxon-polyaxon-operator","polyaxon-polyaxon-gateway","polyaxon-polyaxon-api"]
        # TODO changing to list_namespaced_deployments?
        for e in w.stream(c.list_namespaced_pod, namespace="polyaxon"):
            ob = e["object"]          
            
            for name in monitored_pods:

                #checking if it is one of the pods that we want to monitor 
                if name in ob.metadata.name:
                    
                    # Checking if the pod already is runnig and its underlying containers are ready
                    if ob.status.phase == "Running" and ob.status.container_statuses[0].ready: 
                        log.info(f'{ob.metadata.name} is ready')
                        monitored_pods.remove(name)
                        deployed = deployed + 1

                        #if all monitored pods are running the deployment process was ended
                        if(deployed == 4 ):
                            w.stop()
                            log.info("Finished deploying crucial pods")
                            

        
  

        # Starting post forwarding to the polyaxon api in the background
        log.info("Starting post-forward to polyaxon api:")
        self.post_forward_process = subprocess.Popen("kubectl port-forward  svc/polyaxon-polyaxon-api 31833:80  -n polyaxon",shell=True,stdout=subprocess.PIPE)




      
        
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
            "trialParameters":"${trialParameters.learningRate}",
            "metrics_ip": self.metrics_ip,
        }

        #loading and filling the template
        with open(path.join(path.dirname(__file__), "experiment_template.yaml"), "r") as f:
            job_template = Template(f.read())
            job_yml_objects = job_template.substitute(experiment_definition)
            
        #writing the experiment definition into the file        
        with open(path.join(path.dirname(__file__), self.experiment_file_name), "w") as f:
            f.write(job_yml_objects)
        log.info("Experiment yaml created")
      
        
        if self.create_clean_image:
            log.info("Creating task docker image")   
            #creating docker image inside of the minikube   
            self.image_builder = builder_from_string("minikube")()
            PROJECT_ROOT = os.path.abspath(os.path.join(__file__ ,"../../../"))
            log.info(PROJECT_ROOT)
            self.image_builder.deploy_image(
            f'experiments/polyaxon_minikube/{self.trial_tag}/Dockerfile', self.trial_tag,PROJECT_ROOT)
            print(f"Image: {self.trial_tag}")


        
            
        

    def run(self):

        #TODO add error handling.
        #TODO change to invocking procject comand instead of sending the http request to the polaxon api? 
        log.info("Creating new project:")
        # project = requests.post(f'{self.polyaxon_addr}/api/v1/default/projects/create', json={"name": self.study_name, })

         
        options = f'--name {self.study_name} --description '.split()
        # adding the project description as the last argument  
        options.append(f'{self.project_description}')
        #invoking polyaxon project create comand
        res = self.cli_runner.invoke(create,options)
        log.info(res.output)


      
      
        log.info("Starting polyaxon experiment:")
        #invoking polyaxon run comand with following options
        options = f'-f ./{self.experiment_file_name} --project {self.study_name} --eager'.split()
        res = self.cli_runner.invoke(run,options)
        log.info(res.output)

        
        
        #TODO switch to kubernetes api for monitoring runing trials  
        log.info("Waiting for the run to finish:")
        finished = False
        while not finished:
            runs = self.get_succeeded_runs()
            log.info(f'{runs["count"]} jobs out of {self.jobsCount} succeded')
            
            #checking if all runs were finished
            finished = runs["count"] == self.jobsCount
            sleep(1)
    
        return




    def collect_benchmark_metrics(self):

        log.info("Collecting run results:")
        result = self.get_succeeded_runs()
        log.info(json.dumps(result,indent=4))               
               
        return result["results"]


    def get_succeeded_runs(self, sort_by="duration"):
        
        #TODO add error handling acording to polyaxon api 
        res = requests.get(f'{self.polyaxon_addr}/api/v1/default/{self.study_name}/runs?query=status:succeeded&sort={sort_by}') 
        result = json.loads(res.text)
        return result

    def collect_run_results(self):
        

        log.info("Collecting run results:")
        result = self.get_succeeded_runs()
        log.info(json.dumps(result,indent=4))               
               
        # log.info("\n Experiment finished with following optimal trial:")
        # log.info(result["results"][0])
        return result["results"]
    
    def test(self):
        return super().test()

    def undeploy(self):
        
        if(self.post_forward_process):
            log.info("Terminating post  forwarding process:")
            process = psutil.Process(self.post_forward_process.pid)
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()



        log.info("Undeploying polyaxon:")
        res = self.cli_runner.invoke(teardown,["--yes"])
        #by teardown comand the polyaxon cli doesnt set exit_code if there are some problems
        if("Polyaxon could not teardown the deployment" in res.output):
            raise Exception(f'Exit code: {res.exit_code}  Error message: \n{res.output}')
        elif(res.exit_code == 0):
            print(res.exit_code)
            log.info(res.output)
        else:
            raise Exception(f'Exit code: {res.exit_code}  Error message: \n{res.output}')
   
             
    


        # Waiting untill all polyaxon pods get terminated 
        #TODO add logic in case of no existent polyaxon deployment 
        config.load_kube_config()
        w = watch.Watch()
        c = client.CoreV1Api()
        deployed = 0
        log.info("Waiting for polyaxon pods to be terminated:")
        for e in w.stream(c.list_namespaced_pod, namespace=self.namespace):
            ob = e["object"]
               
            log.debug(f'{deployed} pods out of 4 were killed')
            log.debug("\n new in stream:\n")
            log.debug(ob.metadata.name,ob.status.phase)

            if not ob.status.container_statuses[0].ready:
                log.info(f'Containers of {ob.metadata.name} are terminated')
                deployed = deployed + 1
                if(deployed == 4 ):
                    w.stop()
                    # log.info("Finished ")
                    break
        
      
        log.info("Killed all pods deleteing the namespace:")
        res = c.delete_namespace_with_http_info(name=self.namespace)
        

        #TODO somehow handel the timouts?
        log.info("Checking status of the deleted namespace:")  
        for e in w.stream(c.list_namespace):
            ob = e["object"]
            # if the status of our namespace was changed we check if it the namespace was really removed from the cluster by requesting and expecting it to be not found
            #TODO do this in other way

            
            if ob.metadata.name == self.namespace:
                try:
                    log.debug(c.read_namespace_status_with_http_info(name=self.namespace))
                except ApiException as err:
                    log.info(err)
                    log.info("Namespace sucessfully deleted")
                    w.stop()
                    break


        log.info("Deleting image from minikube")
        self.image_builder.cleanup(self.trial_tag)
        log.info("Finished undeploying")


if __name__ == "__main__":
    #main()
    # bench = PolyaxonBenchmark(resources={
    #         # "dockerUserLogin":"",
    #         # "dockerUserPassword":"",
    #     # "studyName":""
    #     "jobsCount":5,
    #     "workerCount":5,
    #     "loggingLevel":log.INFO,
    #     "dockerImageTag":"nowe",

    #     "metricsIP": urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip()
    #     })
    
    # bench.deploy() 
    # bench.setup()
    # bench.run()
    # # bench.collect_run_results()

    # bench.undeploy()
    # polyaxon config set --host=http://localhost:8000

    
    resources={
        # "studyName":"",
        "dockerImageTag":"task_light",
        "jobsCount":5,
        
        "workerCount":5,
        "loggingLevel":log.INFO,
        "metricsIP": urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip(),
        "createCleanImage":True
    }
    from ml_benchmark.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        benchmark_cls=PolyaxonBenchmark, resources=resources)
    runner.run()

    
    # print(f'polyaxon run -f ./ --project  --eager')
    # print(f'polyaxon run -f ./grid --project --eager'.split())
    # runner = CliRunner()
    # print("start")
    # res = runner.invoke(run,["-f ./grid.yaml --eager -o json"])
    # print("fin")
    # print(res.output,res.exit_code)

    # res = runner.invoke(teardown,["--yes"])    
    # if("Polyaxon could not teardown the deployment" in res.output):
    #     print("Ja pierdole")
    # print(res.output,res.exit_code,res.exception)
    # print("stop")
    # res = runner.invoke(deploy)    
    # print(res.output,res.exit_code)

   # print(res)

