from os import path
import os
from shutil import ExecError
import sys
from time import sleep
from urllib.request import urlopen
from kubernetes.client.rest import ApiException
import random
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from string import Template
import yaml
import docker
import logging as log

from ml_benchmark.benchmark_runner import Benchmark
from ml_benchmark.utils.image_build_wrapper import builder_from_string




class KatibBenchmark(Benchmark):

    def __init__(self, resources) -> None:
        self.resources = resources
        self.group="kubeflow.org"
        self.version="v1beta1"
        self.namespace="kubeflow"
        self.plural="experiments"
        self.experiment_file_name = "grid.yaml"
        self.metrics_ip = resources.get("metricsIP")
        self.generate_new_docker_image = resources.get("generateNewDockerImage",True)     
        self.clean_up = self.resources.get("cleanUp",True)
        self.trial_tag = resources.get("dockerImageTag","mnist_task")
        self.study_name= resources.get("studyName",f'katib-study-{random.randint(0, 100)}')
        self.workerCpu = resources.get("workerCpu",2)
        self.workerMemory= resources.get("workerMemory",2)
        self.workerCount = resources.get("workerCount",5)
        self.jobsCount = resources.get("jobsCount",6)

        self.logging_level= self.resources.get("loggingLevel",log.INFO)
        log.basicConfig(format='%(asctime)s Katib Benchmark %(levelname)s: %(message)s',level=self.logging_level)

        


    
    def deploy(self):
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """


        log.info("Deploying katib:")
        res = os.popen('kubectl apply -k "github.com/kubeflow/katib.git/manifests/v1beta1/installs/katib-standalone?ref=master"').read()
        log.info(res)



        config.load_kube_config()
        w = watch.Watch()
        c = client.CoreV1Api()
        deployed = 0
        log.info("Waiting for all Katib pods to be ready:")
        # From all pods that polyaxon starts we are onlly really intrested for following 4 that are crucial for runnig of the experiments 
        monitored_pods = ["katib-cert-generator","katib-db-manager","katib-mysql","katib-ui","katib-controller"]
        for e in w.stream(c.list_namespaced_pod, namespace=self.namespace):
            ob = e["object"]          

            for name in monitored_pods:

                #checking if it is one of the pods that we want to monitor 
                if name in ob.metadata.name:

                    # Checking if the pod already is runnig and its underlying containers are ready
                    if ob.status.phase == "Running" and ob.status.container_statuses[0].ready: 
                        log.info(f'{ob.metadata.name} is ready')
                        monitored_pods.remove(name)
                        deployed = deployed + 1

                    # Checking for status of cert generator  
                    elif name == "katib-cert-generator" and ob.status.phase == "Succeeded": 
                        log.info(f'{ob.metadata.name} is Succeeded')
                        monitored_pods.remove(name)


                    #if all monitored pods are running the deployment process was ended
                    if not monitored_pods:

                        w.stop()
                        log.info("Finished deploying crucial pods")


        



      
        
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
            "metrics_ip":self.metrics_ip
        }

        #loading and fulling the template
        with open(path.join(path.dirname(__file__), "experiment_template.yaml"), "r") as f:
            job_template = Template(f.read())
            job_yml_objects = job_template.substitute(experiment_definition)
            
        #writing the experiment definition into the file        
        with open(path.join(path.dirname(__file__), self.experiment_file_name), "w") as f:
            f.write(job_yml_objects)
        log.info("Experiment yaml created")
      
      
       #only generating the docker image if specified so.
        if self.generate_new_docker_image:
            log.info("Creating task docker image")   
            #creating docker image inside of the minikube   
            self.image_builder = builder_from_string("minikube")()
            PROJECT_ROOT = os.path.abspath(os.path.join(__file__ ,"../../../"))
            res = self.image_builder.deploy_image(
            f'experiments/katib_minikube/{self.trial_tag}/Dockerfile', self.trial_tag,PROJECT_ROOT)
            log.info(res)
            log.info(f"Image: {self.trial_tag}")  
        
        

    def run(self):
        
        log.info("Starting Katib experiment:")
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
                log.info("Succses: Experiment started")
                log.info(api_response)  
            except ApiException as e:
                log.info("Exception when calling CustomObjectsApi->create_cluster_custom_object: %s\n" % e)
            
        
        #Blocking untill the run is finished
        #The GET /apis/{group}/{version}/namespaces/{namespace}/{plural}/{name} endpoint doesnt support watch argument.
       
        #TODO changing to watching of the jobs instead of the experiment crd?
        experiment = self.get_experiment()
        while "status" not in experiment:
            log.info("Waitinng for the status")
            sleep(5)
            experiment = self.get_experiment()
            # log.info(experiment)
        
    
        while experiment["status"]["conditions"][-1]["type"]!="Succeeded":
            experiment = self.get_experiment()
            succeeded =  experiment["status"].get("trialsSucceeded",0)
            
            log.info(f'Status: {experiment["status"]["conditions"][-1]["reason"]} {succeeded} trials of {self.jobsCount} succeeded')
            log.debug(experiment["status"])
            
            sleep(2)
            
    
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
       
            return resource
        except ApiException as e:
            log.info("Exception when calling CustomObjectsApi->get_namespaced_custom_object_status: %s\n" % e)
            return ""


    def collect_run_results(self):
        
        log.info("Collecting run results:")
           
        experiment = self.get_experiment() 
        
        log.debug("\n Experiment finished with following optimal trial:")
        log.debug(experiment["status"]["currentOptimalTrial"])
        return experiment["status"]["currentOptimalTrial"] 
        
    
    def test(self):
        return super().test()

    def undeploy(self):
              
        log.info("Deleteing the experiment crd from the cluster")
        config.load_kube_config()
        api = client.CustomObjectsApi()
        try:
            #deleting all experiment crds 
            resource = api.delete_collection_namespaced_custom_object(
                    group=self.group,
                    version=self.version,
                    namespace=self.namespace,
                    plural=self.plural,
                )
            log.info(resource)
       
        except ApiException as e:
            log.info("Exception when calling CustomObjectsApi->get_namespaced_custom_object_status: %s\n" % e)



        w = watch.Watch()
        c = client.CoreV1Api()
        log.info("Deleteing the namespace:")
        res = c.delete_namespace_with_http_info(name=self.namespace)    
        

        log.info("Checking status of the  namespace:")
        #if the namespace was still existent we must wait till it is really terminated
        for e in w.stream(c.list_namespace):
            ob = e["object"]
            # if the status of our namespace was changed we check if it the namespace was really removed from the cluster by requesting and expecting it to be not found
            if ob.metadata.name == self.namespace:
                try:
                    log.debug(c.read_namespace_status_with_http_info(name=self.namespace))
                except ApiException as err:
                    
                    if self.clean_up:
                        log.info("Deleteing task docker image from minikube")
                        sleep(2)
                        self.image_builder.cleanup(self.trial_tag)
                    
                    
                    if(err.status != 404):
                        raise Exception("Something went wrong",err)
                    else: 
                        log.info("Namespace sucessfully deleted")
                        w.stop()


        log.info("Finished undeploying")

      



if __name__ == "__main__":
    #main()

    resources={
            # "dockerUserLogin":"",
            # "dockerUserPassword":"",
            # "studyName":""
            "jobsCount":2,
            # "dockerImageTag":"light_task",
            "workerCount":2,
            "metricsIP": urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip(),
            "generateNewDockerImage": True,
            "cleanUp": True ,
            }
    # bench = KatibBenchmark(resources=resources)
    # bench.deploy()
    # bench.setup()
    # bench.run()
    # bench.collect_run_results()
    # bench.undeploy()



    from ml_benchmark.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        benchmark_cls=KatibBenchmark, resources=resources)
    runner.run()