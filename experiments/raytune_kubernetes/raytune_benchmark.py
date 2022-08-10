from os import path
import subprocess
import time

import libtmux

import ray
from ray import tune

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
        self.workerMemory = resources.get("workerMem", 1)
        self.workerCount = resources.get("workerCount", 1)
        self.metrics_ip = resources.get("metricsIP")

        # TMUX session for ray cluster port forward
        self.ray_pf_tmux_session = libtmux.Server().find_where({"session_name": "ray-pf"})
        self.ray_pf_tmux_window = self.ray_pf_tmux_session.new_window(attach=False, window_name="ray-pf")

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
                subprocess.run(["kubectl", "-n", self.namespace, "get", "pod"])
                print("\n")
                last_report_t = curr_report_t 
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            ray_controller_pods = [ pod for pod in pods if ("ray-operator" in pod.metadata.name) or ("ray-cluster-ray-head-type" in pod.metadata.name)]
            ray_worker_pods = [ pod for pod in pods if "ray-cluster-ray-worker-type" in pod.metadata.name ]
            if len(ray_controller_pods) != 2 or len(ray_worker_pods) != self.workerCount:
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
            "metrics_ip": self.metrics_ip,
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
        # raycluster port forward
        ray_pf_tmux_pane = self.ray_pf_tmux_window.panes[0];
        ray_pf_tmux_pane.send_keys(f"kubectl -n {self.namespace} port-forward service/ray-cluster-ray-head 10001:10001")
        
        # init ray
        ray.init("ray://127.0.0.1:10001")

    def run(self):
        """
            Executing the hyperparameter optimization on the deployed platfrom.
            use the metrics object to collect and store all measurments on the workers.
        """
        def raytune_func(config):
            """The function for training and validation, that is used for hyperparameter optimization.
            Beware Ray Synchronisation: https://docs.ray.io/en/latest/tune/user-guide.html

            Args:
                config ([type]): [description]
            """
            objective = config.get("objective")
            hyperparameters = config.get("hyperparameters")
            objective.set_hyperparameters(hyperparameters)
            # these are the results, that can be used for the hyperparameter search
            objective.train()
            validation_scores = objective.validate()
            tune.report(
                macro_f1_score=validation_scores["macro avg"]["f1-score"])

        grid = dict(
            input_size=28*28, learning_rate=tune.grid_search([1e-4]),
            weight_decay=1e-6,
            hidden_layer_config=tune.grid_search([[20], [10, 10]]),
            output_size=10)
        task = MnistTask(config_init={"epochs": 1})
        self.analysis = tune.run(
            raytune_func,
            config=dict(
                objective=task.create_objective(),
                hyperparameters=grid,
            ),
            sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
            ),
            local_dir="~/ray_results",
            resources_per_trial={"cpu": self.workerCpu}
        )

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
    
    def _undeploy_watch(self) -> None:
        last_report_t = None
        while True:
            curr_report_t = time.time()
            if last_report_t is None or (curr_report_t - last_report_t > 5):
                subprocess.run(["kubectl", "-n", self.namespace, "get", "pod"])
                print("\n")
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

        # delete port-forward tmux window
        self.ray_pf_tmux_window.kill_window()




if __name__ == "__main__":
    from ml_benchmark.benchmark_runner import BenchmarkRunner

    resource_definition = {
        "kubernetesNamespace": "st-hpo",
        "workerCpu": 1,
        "workerMemory": 1,
        "workerCount": 1,
        "metricsIP": "192.168.0.101"
    }

    runner = BenchmarkRunner(
        benchmark_cls=RaytuneBenchmark, resources=resource_definition)
    runner.run()
