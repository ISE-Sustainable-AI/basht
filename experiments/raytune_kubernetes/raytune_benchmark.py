from os import path
import subprocess
import argparse

import ray
from ray import tune
from ray.tune import Stopper

from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.utils import create_from_yaml, FailToCreateError

from basht.benchmark_runner import Benchmark
from basht.workload.objective import Objective
from basht.utils.yaml import YamlTemplateFiller


class TrialStopper(Stopper):
    def __call__(self, trial_id, result):
        return result['time_total_s'] > 60*15

    def stop_all(self):
        return False


class RaytuneBenchmark(Benchmark):

    def __init__(self, resources) -> None:
        self.namespace = resources.get("kubernetesNamespace", "st-hpo")
        self.workerCpu = resources.get("workerCpu", 1)
        self.workerMemory = resources.get("workerMemory", 1)
        self.workerCount = resources.get("workerCount", 1)
        self.metricsIP = resources.get("metricsIP")
        self.nfsServer = resources.get("nfsServer")
        self.nfsPath = resources.get("nfsPath")

        self.grid = resources.get("hyperparameter")

        # K8s setup
        config.load_kube_config()
        self.k8s_api_client = client.ApiClient()
        self.k8s_custom_objects_api = client.CustomObjectsApi()
        self.k8s_core_v1_api = client.CoreV1Api()
        self.k8s_apps_v1_api = client.AppsV1Api()

    def _deploy_watch(self) -> None:
        w = watch.Watch()
        ray_head_is_ready = False
        for event in w.stream(self.k8s_core_v1_api.list_namespaced_pod, namespace=self.namespace):
            print(f"Event: {event['type']} {event['object'].metadata.name} {event['object'].status.phase}")
            if "ray-head" in event['object'].metadata.name and event['object'].metadata.labels['ray-node-status'] == "up-to-date":
                ray_head_is_ready = True
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            ray_pods = [pod for pod in pods if "ray-operator" in pod.metadata.name or "ray-cluster" in pod.metadata.name]
            if len(ray_pods) != self.workerCount + 1:
                continue
            else:
                running_ray_pods = [pod for pod in ray_pods if pod.status.phase == "Running"]
                if len(running_ray_pods) == 1 + self.workerCount and ray_head_is_ready:
                    w.stop()
        print("Ray pods are ready")

    def deploy(self) -> None:
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """
        # deploy ray operator
        try:
            create_from_yaml(
                self.k8s_api_client,
                path.join(path.dirname(__file__), "ray-template/ray-operator.yaml"),
                namespace=self.namespace, verbose=True
            )
        except FailToCreateError as e:
            raise e

        # deploy ray cluster
        ray_cluster_definition = {
            "ray_worker_num": self.workerCount - 1,
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
                ["kubectl", "-n", self.namespace, "port-forward",
                    "service/ray-cluster-ray-head", "10001:10001"],
                stdout=pf_log
            )
        ray.init("ray://localhost:10001")

    def run(self):
        """
            Executing the hyperparameter optimization on the deployed platfrom.
            use the metrics object to collect and store all measurments on the workers.
        """
        grid = self.grid

        def raytune_func(config, checkpoint_dir=None):
            hyperparameter = config.get("hyperparameters")

            objective = Objective(
                dl_framework=self.workload.get("dl_framework"), model_cls=self.workload.get("model_cls"),
                epochs=self.workload.get("epochs"), device=self.workload.get("device"),
                task=self.workload.get("task"), hyperparameter=hyperparameter)
            # these are the results, that can be used for the hyperparameter search
            objective.load()
            objective.train()
            validation_scores = objective.validate()
            tune.report(
                macro_f1_score=validation_scores["macro avg"]["f1-score"])

        self.analysis = tune.run(
            raytune_func,
            config=dict(
                hyperparameters=grid,
            ),
            sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
            ),
            local_dir="/home/ray/ray-results",
            resources_per_trial={"cpu": self.workerCpu},
            stop=TrialStopper()
        )

        print(self.analysis.get_best_config(
            metric="macro_f1_score", mode="max")["hyperparameters"])
        return

    def collect_run_results(self):
        self.best_hyp_config = self.analysis.get_best_config(
            metric="macro_f1_score", mode="max")["hyperparameters"]

    def test(self):
        # evaluating and retrieving the best model to generate test results.
        hyperparameter = self.best_hyp_config

        objective = Objective(
            dl_framework=self.workload.get("dl_framework"), model_cls=self.workload.get("model_cls"),
            epochs=self.workload.get("epochs"), device=self.workload.get("device"),
            task=self.workload.get("task"), hyperparameter=hyperparameter)
        self.training_loss = objective.train()
        self.test_scores = objective.test()

    def collect_benchmark_metrics(self):
        """
            Describes the collection of all gathered metrics, which are not used by the HPO framework
            (Latencies, CPU Resources, etc.). This step runs outside of the HPO Framework.
            Ensure to optain all metrics loggs and combine into the metrics object.
        """
        results = dict(
            test_scores=self.test_scores,
            training_loss=self.training_loss,
        )

        return results

    def _undeploy_watch_ray_cluster(self):
        w = watch.Watch()
        for event in w.stream(
            self.k8s_custom_objects_api.list_namespaced_custom_object,
                group="cluster.ray.io", version="v1", namespace=self.namespace, plural="rayclusters"):
            print(f"Event: {event['type']} {event['object']['kind']} {event['object']['status']['phase']}")
            if event['type'] == "DELETED":
                w.stop()
        print("Ray Cluster was deleted successfully")

    def _undeploy_watch_ray_operator(self):
        w = watch.Watch()
        for event in w.stream(self.k8s_core_v1_api.list_namespaced_pod, namespace=self.namespace):
            print(
                f"Event: {event['type']} {event['object'].metadata.name} {event['object'].status.phase}")
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            ray_operator_pods = [
                pod for pod in pods if "ray-operator" in pod.metadata.name]
            if len(ray_operator_pods) != 0:
                continue
            else:
                w.stop()
        print("Ray Operator pod was deleted successfully")

    def undeploy(self):
        """
            The clean-up procedure to undeploy all components of the HPO Framework that were deployed in the
            Deploy step.
        """
        ray.shutdown()
        self.portforward_proc.terminate()

        # undeploy ray cluster
        try:
            self.k8s_custom_objects_api.delete_namespaced_custom_object(
                group="cluster.ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                name="ray-cluster"
            )
        except ApiException as e:
            raise e

        self._undeploy_watch_ray_cluster()

        # undeploy ray operator
        try:
            self.k8s_apps_v1_api.delete_namespaced_deployment(
                "ray-operator", self.namespace)
        except ApiException as e:
            raise e

        self._undeploy_watch_ray_operator()


def create_ray_grid(grid):
    ray_grid = {}
    for key, value in grid.items():
        if type(grid[key]) is list:
            ray_grid[key] = tune.grid_search(value)
        else:
            ray_grid[key] = value
    return dict(ray_grid)


if __name__ == "__main__":
    from basht.benchmark_runner import BenchmarkRunner
    from basht.utils.yaml import YMLHandler
    import os
    from basht.config import Path

    yml_path = os.path.join(Path.root, "experiments/raytune_kubernetes/resource_definition.yml")
    resource_definition = YMLHandler.load_yaml("resource_definition.yml")

    runner = BenchmarkRunner(
        benchmark_cls=RaytuneBenchmark, resources=resource_definition)

    runner.run()
