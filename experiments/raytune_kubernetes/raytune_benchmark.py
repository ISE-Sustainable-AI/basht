from os import path
import logging
from time import sleep
import ray
from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.utils import FailToCreateError, create_from_yaml
from ray import tune
from ray.tune import Stopper

from basht.benchmark_runner import Benchmark
from basht.config import Path
from basht.utils.generate_grid_search_space import generate_grid_search_space
from basht.utils.yaml import YamlTemplateFiller, YMLHandler
from basht.workload.objective import Objective


class TrialStopper(Stopper):
    def __call__(self, trial_id, result):
        return result['time_total_s'] > 60*15

    def stop_all(self):
        return False

log = logging.getLogger('RaytuneBenchmark')
log.setLevel(logging.DEBUG)


class RaytuneBenchmark(Benchmark):
    _path = path.dirname(__file__)

    def __init__(self, resources) -> None:
        self.namespace = resources.get("kubernetesNamespace")
        self.workerCpu = resources.get("workerCpu")
        self.workerMemory = resources.get("workerMemory")
        self.workerCount = resources.get("workerCount")
        self.metricsIP = resources.get("metricsIP")
        self.kubernetes_master_ip = resources.get("kubernetesMasterIP")
        self.ray_node_port = resources.get("rayNodePort")
        self.search_space = generate_grid_search_space(resources.get("hyperparameter"))
        self.delete_after_run = resources.get("deleteAfterRun")
        self.workload = resources.get("workload")
        self.storageClass = resources.get("kubernetesStorageClass")
        self.docker_image_tag = resources.get("dockerImageTag")
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
            log.debug(f"Event: {event['type']} {event['object'].metadata.name} {event['object'].status.phase}")
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
        log.info("Ray pods are ready")

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
            log.info("Namespace created. status='%s'" % str(resp))
        except ApiException as e:
            if self._is_create_conflict(e):
                log.info("Deployment (namespace) already exists")
            else:
                raise e
        # create serviceaccount
        try:
            body = {"metadata": {"name": "ray-operator-serviceaccount"}}
            client.CoreV1Api().create_namespaced_service_account(self.namespace, body)
            log.info("ServiceAccount created")
        except ApiException as e:
            if self._is_create_conflict(e):
                log.info("Service Account  already exists")
            else:
                log.error("Service Account was not created.")
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
            log.info("Role and RoleBinding created")
        except ApiException as e:
            if self._is_create_conflict(e):
                log.info("role or role binding already exists")
            else:
                log.error("failed to create role or role binding.")
                raise e
        # create resource definition
        try:
            resource_def = YMLHandler.load_yaml(
                path.join(
                    Path.root_path, "experiments/raytune_kubernetes/ray-template/preliminaries/cluster_crd.yaml"))
            client.ApiextensionsV1Api().create_custom_resource_definition(body=resource_def)
            log.info("Ray CustomResourceDefinition created")
        except ApiException as e:
            if self._is_create_conflict(e):
                log.info("CRD already exists")
            else:
                raise e

        try:
            pv = YamlTemplateFiller.load_and_fill_yaml_template(
                path.join(Path.root_path, "experiments/raytune_kubernetes/ray-template/preliminaries/pv.yaml"),
                {"storageClass":self.storageClass}
            )
            self.k8s_core_v1_api.create_namespaced_persistent_volume_claim(
                namespace=self.namespace,
                body=next(pv) # XXX we must fix load_and_fill_yaml_template
            )
            log.info("PersistentVolumeClaim created")
        except ApiException as e:
            if self._is_create_conflict(e):
                log.info("persistent volume exists, might contain data from previous runs")
            else:
                log.error("failed to create persistent volume needed for ray coordinator")
                raise e

        # deploy ray operator
        try:
            create_from_yaml(
                self.k8s_api_client,
                path.join(path.dirname(__file__), "ray-template/ray-operator.yaml"),
                namespace=self.namespace, verbose=True
            )
            log.info("Ray Operator created")
        except FailToCreateError as e:
            if self._is_create_conflict(e):
                log.error("Deployment (operator) already exists")
            else:
                raise e
        # deploy ray cluster
        ray_cluster_definition = {
            "ray_worker_num": self.workerCount - 1,
            "worker_cpu": self.workerCpu,
            "worker_mem": f"{self.workerMemory}Gi",
            "metrics_ip": self.metricsIP,
            "RAY_HEAD_IP": "$RAY_HEAD_IP",
            "docker_image": self.docker_image_tag
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
                body=ray_cluster_json_objects)
            log.info("Ray Cluster created")
        except FailToCreateError as e:
            log.error("failed to create ray cluster object - maybe it already exists")
            raise e
        # wait
        self._deploy_watch()

        try:
            self.k8s_core_v1_api.patch_namespaced_service(
                name="ray-cluster-ray-head",
                namespace=self.namespace,
                body={
                    "spec": {
                        "type": "NodePort",
                        "ports": [
                            {
                                "name": "client",
                                "nodePort": self.ray_node_port,
                                "port": 10001
                            }
                        ]
                    }
                }
            )
        except ApiException as e:
            log.error("failed to update ray coordinator service to expose port")
            raise e

        ray.init(f"ray://{self.kubernetes_master_ip}:{self.ray_node_port}")
        log.info("Ray is ready")

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
        search_space = self.create_ray_grid(self.search_space)
        config = dict(
                hyperparameters=search_space,
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
            log.debug(f"Event: {event['type']} {event['object']['kind']} {event['object']['status']['phase']}")
            if event['type'] == "DELETED":
                w.stop()
        log.info("Ray Cluster was deleted successfully")

    def _undeploy_watch_ray_operator(self):
        w = watch.Watch()
        for event in w.stream(self.k8s_core_v1_api.list_namespaced_pod, namespace=self.namespace):
            log.debug(
                f"Event: {event['type']} {event['object'].metadata.name} {event['object'].status.phase}")
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            ray_operator_pods = [
                pod for pod in pods if "ray-operator" in pod.metadata.name]
            if len(ray_operator_pods) != 0:
                continue
            else:
                w.stop()
        log.info("Ray Operator pod was deleted successfully")

    def undeploy(self):
        """
            The clean-up procedure to undeploy all components of the HPO Framework that were deployed in the
            Deploy step.
        """
        ray.shutdown()

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
            if not RaytuneBenchmark._is_status(e, 404):
                log.error("failed to delete ray cluster object - you might need to manual patch the finalizer to fix this")
                raise e

        self._undeploy_watch_ray_cluster()

        # undeploy ray operator
        try:
            self.k8s_apps_v1_api.delete_namespaced_deployment(
                "ray-operator", self.namespace)
        except ApiException as e:
            if not RaytuneBenchmark._is_status(e, 404):
                log.error("failed to delete ray operator deployment")
                raise e

        self._undeploy_watch_ray_operator()

        try:
            self.k8s_core_v1_api.delete_namespaced_persistent_volume_claim(
                namespace=self.namespace,
                name="ray-results"
            )
            log.info("PersistentVolumeClaim deleted successfully")
        except ApiException as e:
             if not RaytuneBenchmark._is_status(e, 404):
                log.error("failed to delete volume claim, unclean environment")
                raise e


        #XXX this might be considered for the whole thing instead of just the namspace, also we do not want to delete the CRD
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
    def _is_status(e, status):
        if isinstance(e, ApiException):
            if e.status == status:
                return True
        if isinstance(e, FailToCreateError):
            if e.api_exceptions is not None:
                # lets quickly check if all status codes are 409 -> componetnes exist already
                if set(map(lambda x: x.status, e.api_exceptions)) == {status}:
                    return True
        return False


    @staticmethod
    def _is_create_conflict(e):
       return RaytuneBenchmark._is_status(e, 409)

    def _watch_namespace(self):
        c = client.CoreV1Api()
        no_exception = True
        while no_exception:
            try:
                c.read_namespace_status(self.namespace)
                sleep(2)
            except ApiException:
                no_exception = False


def main():
    # from urllib.request import urlopen

    from basht.benchmark_runner import BenchmarkRunner
    # from basht.utils.yaml import YMLHandler

    resource_definition = YMLHandler.load_yaml(path.join(path.dirname(__file__), "resource_definition.yml"))
    # resource_definition["metricsIP"] = urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip()# resource_definition["metricsIP"]
    # resource_definition["kubernetesMasterIP"] = "130.149.158.143"

    runner = BenchmarkRunner(
        benchmark_cls=RaytuneBenchmark, resources=resource_definition)
    runner.run()


if __name__ == "__main__":
    main()
