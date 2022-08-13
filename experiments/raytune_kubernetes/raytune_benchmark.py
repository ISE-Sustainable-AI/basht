from os import path
import os
import subprocess
import time

import ray
from ray import tune
from ray.job_submission import JobSubmissionClient, JobStatus

from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.utils import create_from_yaml, FailToCreateError

from ml_benchmark.benchmark_runner import Benchmark
from ml_benchmark.workload.mnist.mnist_task import MnistTask
from ml_benchmark.config import Path
from ml_benchmark.utils.yaml_template_filler import YamlTemplateFiller


class RaytuneBenchmark(Benchmark):

    def __init__(self, resources) -> None:
        self.namespace = resources.get("kubernetesNamespace", "st-hpo")
        self.workerCpu = resources.get("workerCpu", 1)
        self.workerMemory = resources.get("workerMemory", 1)
        self.workerCount = resources.get("workerCount", 1)
        self.metricsIP = resources.get("metricsIP")
        self.nfsServer = resources.get("nfsServer")
        self.nfsPath = resources.get("nfsPath")

        # K8s setup
        config.load_kube_config()
        self.k8s_api_client = client.ApiClient()
        self.k8s_custom_objects_api = client.CustomObjectsApi()
        self.k8s_core_v1_api = client.CoreV1Api()
        self.k8s_apps_v1_api = client.AppsV1Api()
    
    def _deploy_watch(self) -> None:
        last_report_t = None
        while True:
            curr_report_t = time.time()
            if last_report_t is None or (curr_report_t - last_report_t > 5):
                print("\n")
                subprocess.run(["kubectl", "-n", self.namespace, "get", "pod"])
                last_report_t = curr_report_t 
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            ray_controller_pods = [ pod for pod in pods if ("ray-operator" in pod.metadata.name) or ("ray-cluster-ray-head-type" in pod.metadata.name)]
            ray_worker_pods = [ pod for pod in pods if "ray-cluster-ray-worker-type" in pod.metadata.name ]
            if len(ray_controller_pods) < 2 or len(ray_worker_pods) < self.workerCount:
                continue
            else:
                running_ray_controller_pods = [ pod for pod in ray_controller_pods if pod.status.phase == "Running" ]
                running_ray_worker_pods = [ pod for pod in ray_worker_pods if pod.status.phase == "Running" ]
                if len(running_ray_controller_pods) != 2 or len(running_ray_worker_pods) != self.workerCount:
                    continue
            break
        
        print("Ray pods are ready")

    def deploy(self) -> None:
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """
        # deploy ray operator
        try:
            resp = create_from_yaml(
                self.k8s_api_client,
                path.join(path.dirname(__file__), "ray-template/ray-operator.yaml"),
                namespace=self.namespace, verbose=True
            )
        except FailToCreateError as e:
            raise e
        
        # deploy ray cluster
        ray_cluster_definition = {
            "worker_num": self.workerCount,
            "worker_cpu": self.workerCpu,
            "worker_mem": f"{self.workerMemory}Gi",
            "metrics_ip": self.metricsIP,
            "nfs_server": self.nfsServer,
            "nfs_path": self.nfsPath,
            "RAY_HEAD_IP": "$RAY_HEAD_IP"
        }

        ray_cluster_yml_objects = YamlTemplateFiller.load_and_fill_yaml_template(
            path.join(path.dirname(__file__), "ray-template/ray-cluster-template.yaml"),
            ray_cluster_definition
        )

        ray_cluster_json_objects = next(ray_cluster_yml_objects)
        try:
            self.k8s_custom_objects_api.create_namespaced_custom_object(
                group="cluster.ray.io", 
                version="v1", 
                namespace=self.namespace, 
                plural="rayclusters",
                body=ray_cluster_json_objects
            )
        except FailToCreateError as e:
            raise e

        self._deploy_watch()
    
    def setup(self):
        with open("portforward_log.txt", 'w') as pf_log:
            self.portforward_proc = subprocess.Popen(
                ["kubectl", "-n", self.namespace, "port-forward", "service/ray-cluster-ray-head", "10001:10001"],
                stdout=pf_log
            )
        ray.init("ray://localhost:10001")

    #def _run_watch(self):
    #    while True:
    #        status = self.ray_job_client.get_job_status(self.job_id)
    #        print(f"Status: {status}")
    #        with open("./log.txt", "w") as log_file:
    #            log_file.write(self.ray_job_client.get_job_logs(self.job_id))
    #        if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
    #            break
    #        time.sleep(2)

    def run(self):
        """
            Executing the hyperparameter optimization on the deployed platfrom.
            use the metrics object to collect and store all measurments on the workers.
        """
        #self.ray_job_client = JobSubmissionClient("http://127.0.0.1:8265")
        #self.job_id = self.ray_job_client.submit_job(
        #    entrypoint="python script.py",
        #    runtime_env={
        #        "working_dir": "./tune-env"
        #    }
        #)
        #self._run_watch()


        grid = dict(
            input_size=28*28, learning_rate=tune.grid_search([1e-4, 0.1]),
            weight_decay=1e-6,
            hidden_layer_config=tune.grid_search([[20], [10, 10]]),
            output_size=10)
        
        task = MnistTask(config_init={"epochs": 1})

        def raytune_func(config, checkpoint_dir=None):
            import ray
            #from ml_benchmark.workload.mnist.mnist_task import MnistTask
            
            #task = MnistTask(config_init={"epochs": 1})
            objective = ray.get(config.get("objective"))
            #objective = task.create_objective()

            hyperparameters = config.get("hyperparameters")
            objective.set_hyperparameters(hyperparameters)
            # these are the results, that can be used for the hyperparameter search
            objective.train()
            validation_scores = objective.validate()
            tune.report(
                macro_f1_score=validation_scores["macro avg"]["f1-score"])

        
        objective_ref = ray.put(task.create_objective())
        
        self.analysis = tune.run(
            raytune_func,
            config=dict(
                objective=objective_ref,
                hyperparameters=grid,
            ),
            sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
            ),
            local_dir="/home/ray/ray-results",
            resources_per_trial={"cpu": 1}
        )

        print(self.analysis.get_best_config(metric="macro_f1_score", mode="max")["hyperparameters"])
        
        return

    def collect_run_results(self):
        return 
        self.best_hyp_config = self.analysis.get_best_config(
            metric="macro_f1_score", mode="max")["hyperparameters"]

    def test(self):
        return
        # evaluating and retrieving the best model to generate test results.
        task = MnistTask(config_init={"epochs": 1})
        objective = task.create_objective()
        objective.set_hyperparameters(self.best_hyp_config)
        self.training_loss = objective.train()
        self.test_scores = objective.test()

    def collect_benchmark_metrics(self):
        """
            Describes the collection of all gathered metrics, which are not used by the HPO framework
            (Latencies, CPU Resources, etc.). This step runs outside of the HPO Framework.
            Ensure to optain all metrics loggs and combine into the metrics object.
        """
        return
        results = dict(
            test_scores=self.test_scores,
            training_loss=self.training_loss,
        )

        return results
    
    def _undeploy_watch(self):
        last_report_t = None
        while True:
            curr_report_t = time.time()
            if last_report_t is None or (curr_report_t - last_report_t > 5):
                print("\n")
                subprocess.run(["kubectl", "-n", self.namespace, "get", "pod"])
                last_report_t = curr_report_t 
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            if len(pods) == 0:
                break
            ray_pods = [ pod for pod in pods if "ray" in pod.metadata.name ]
            if len(ray_pods) != 0:
                continue
            else:
                break
        
        print("Ray pods were deleted successfully")

    def undeploy(self):
        """
            The clean-up procedure to undeploy all components of the HPO Framework that were deployed in the
            Deploy step.
        """
        # undeploy ray cluster
        try:
            self.k8s_custom_objects_api.delete_namespaced_custom_object(
                group="cluster.ray.io", 
                version="v1", 
                namespace=self.namespace, 
                plural="rayclusters",
                name="ray-cluster",
                grace_period_seconds=10
                #body=self.ray_cluster_json_objects
            )
        except ApiException as e:
            raise e
        
        # undeploy ray operator
        try:
            self.k8s_apps_v1_api.delete_namespaced_deployment("ray-operator", self.namespace)
        except ApiException as e:
            raise e      
        
        self._undeploy_watch()

        # terminate port-forward proc
        self.portforward_proc.terminate()




if __name__ == "__main__":
    from ml_benchmark.benchmark_runner import BenchmarkRunner
    import json
    
    with open("resource_definition.json") as res_def_file:
        resource_definition = json.load(res_def_file)

    runner = BenchmarkRunner(
        benchmark_cls=RaytuneBenchmark, resources=resource_definition)
    runner.run()
