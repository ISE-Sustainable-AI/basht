from os import path
import subprocess
import time

import ray
from ray import tune
from ray.tune import Stopper

from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.utils import create_from_yaml, FailToCreateError

from basht.benchmark_runner import Benchmark
from basht.workload.objective import Objective
from basht.utils.yaml import YamlTemplateFiller, YMLHandler
from basht.utils.generate_grid_search_space import generate_grid_search_space
from basht.config import Path


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
        self.delete_after_run = resources.get("deleteAfterRun")
        self.workload = resources.get("workload")

        # K8s setup
        config.load_kube_config(context=resources.get("kubernetesContext"))
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
        # create the namespace
        try:

            resp = client.CoreV1Api().create_namespace(
                client.V1Namespace(metadata=client.V1ObjectMeta(name=self.namespace)))
            print("Namespace created. status='%s'" % str(resp))
        except ApiException as e:
            if self._is_create_conflict(e):
                print("Deployment (namespace) already exists")
            else:
                raise e
        # create serviceaccount
        try:
            body = {"metadata": {"name": "ray-operator-serviceaccount"}}
            client.CoreV1Api().create_namespaced_service_account(self.namespace, body)
        except ApiException as e:
            print("Service Account was not created.")
            raise e
        # create roles
        try:
            role = YMLHandler.load_yaml(
                path.join(
                    Path.root_path, "experiments/raytune_kubernetes/ray-template/preliminaries/role.yaml"))
            role_binding = YMLHandler.load_yaml(
                path.join(
                    Path.root_path, "experiments/raytune_kubernetes/ray-template/preliminaries/role-binding.yaml"))
            client.RbacAuthorizationV1Api().create_namespaced_role(self.namespace, body=role)
            client.RbacAuthorizationV1Api().create_namespaced_role_binding(self.namespace, body=role_binding)
        except ApiException as e:
            print("Role could not be created")
            raise e
        # create resource definition
        try:
            resource_def = YMLHandler.load_yaml(
                path.join(
                    Path.root_path, "experiments/raytune_kubernetes/ray-template/preliminaries/cluster_crd.yaml"))
            client.ApiextensionsV1Api().create_custom_resource_definition(body=resource_def)
        except ApiException as e:
            if self._is_create_conflict(e):
                print("Deployment already exists")
            else:
                raise e

        # deploy ray operator
        try:
            create_from_yaml(
                self.k8s_api_client,
                path.join(path.dirname(__file__), "ray-template/ray-operator.yaml"),
                namespace=self.namespace, verbose=True
            )
        except FailToCreateError as e:
            if self._is_create_conflict(e):
                print("Deployment (operator) already exists")
            else:
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

        # wait for operator to be ready

        time.sleep(30)

        ray_cluster_json_objects = next(ray_cluster_yml_objects)
        try:
            self.k8s_custom_objects_api.create_namespaced_custom_object(
                group="cluster.ray.io",
                version="v1",
                namespace=self.namespace,
                plural="rayclusters",
                body=ray_cluster_json_objects)
        except FailToCreateError as e:
            raise e
        # wait
        self._deploy_watch()

        with open("portforward_log.txt", 'w') as pf_log:
            self.portforward_proc = subprocess.Popen(
                ["kubectl", "-n", self.namespace, "port-forward",
                    "service/ray-cluster-ray-head", "10001:10001"],
                stdout=pf_log
            )

        ray.init("ray://localhost:10001")

    def setup(self):
        return

    @staticmethod
    def raytune_func(config, checkpoint_dir=None):
        hyperparameter = config.get("hyperparameters")
        workload = config.get("workload")
        objective = Objective(
            dl_framework=workload.get("dl_framework"), model_cls=workload.get("model_cls"),
            epochs=workload.get("epochs"), device=workload.get("device"),
            task=workload.get("task"), hyperparameter=hyperparameter)
        # these are the results, that can be used for the hyperparameter search
        objective.load()
        objective.train()
        validation_scores = objective.validate()
        tune.report(
            macro_f1_score=validation_scores["macro avg"]["f1-score"])

    def run(self):
        """
            Executing the hyperparameter optimization on the deployed platfrom.
            use the metrics object to collect and store all measurments on the workers.
        """
        grid = self.create_ray_grid(self.grid)
        config = dict(
                hyperparameters=grid,
                workload=self.workload
            )
        self.analysis = tune.run(
            RaytuneBenchmark.raytune_func,
            config=config,
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
        objective.load()
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
            group="cluster.ray.io", version="v1",
                namespace=self.namespace, plural="rayclusters"):
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
        if self.delete_after_run:
            client.CoreV1Api().delete_namespace(self.namespace)
            self._watch_namespace()

    @staticmethod
    def create_ray_grid(grid):
        ray_grid = {}
        for key, value in grid.items():
            if isinstance(value, dict):
                value = list(value.values())
            if isinstance(value, list):
                ray_grid[key] = tune.grid_search(value)
            else:
                ray_grid[key] = value
        return ray_grid

    @staticmethod
    def _is_create_conflict(e):
        if isinstance(e, ApiException):
            if e.status == 409:
                return True
        if isinstance(e, FailToCreateError):
            if e.api_exceptions is not None:
                # lets quickly check if all status codes are 409 -> componetnes exist already
                if set(map(lambda x: x.status, e.api_exceptions)) == {409}:
                    return True
        return False


def main():
    from basht.benchmark_runner import BenchmarkRunner
    from urllib.request import urlopen
    from basht.utils.yaml import YMLHandler

    resource_definition = YMLHandler.load_yaml(path.join(path.dirname(__file__), "resource_definition.yml"))
    resource_definition["metricsIP"] = urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip()
    resource_definition["nfsServer"] = resource_definition["metricsIP"]
    resource_definition["hyperparameter"] = generate_grid_search_space(
        resource_definition["hyperparameter"])

    runner = BenchmarkRunner(
        benchmark_cls=RaytuneBenchmark, resources=resource_definition)
    runner.run()


if __name__ == "__main__":
    main()
