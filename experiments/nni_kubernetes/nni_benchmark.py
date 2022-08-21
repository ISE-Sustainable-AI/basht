from os import path
import subprocess
import argparse
import time

from nni.experiment import (
    Experiment,
    FrameworkAttemptCompletionPolicy,
    FrameworkControllerRoleConfig,
    K8sNfsConfig
)

from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.utils import create_from_yaml, FailToCreateError

from ml_benchmark.benchmark_runner import Benchmark
from ml_benchmark.workload.mnist.mnist_task import MnistTask
from ml_benchmark.utils.yaml_template_filler import YamlTemplateFiller

global_grid = None
n_trials = 0

class NNIBenchmark(Benchmark):

    def __init__(self, resources) -> None:
        self.namespace = resources.get("kubernetesNamespace", "st-hpo")
        self.workerCpu = resources.get("workerCpu", 1)
        self.workerMemory = resources.get("workerMemory", 1)
        self.workerCount = resources.get("workerCount", 1)
        self.metricsIP = resources.get("metricsIP")
        self.nfsServer = resources.get("nfsServer")
        self.nfsPath = resources.get("nfsPath")

        self.grid = global_grid
        self.experiment = Experiment('frameworkcontroller')

        # K8s setup
        config.load_kube_config()
        self.k8s_api_client = client.ApiClient()
        self.k8s_custom_objects_api = client.CustomObjectsApi()
        self.k8s_core_v1_api = client.CoreV1Api()
        self.k8s_apps_v1_api = client.AppsV1Api()

    def _deploy_watch(self) -> None:
        w = watch.Watch()
        for event in w.stream(self.k8s_core_v1_api.list_namespaced_pod, namespace=self.namespace):
            print(
                f"Event: {event['type']} {event['object'].metadata.name} {event['object'].status.phase}")
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            fc_pods = [pod for pod in pods if "frameworkcontroller" in pod.metadata.name]
            if len(fc_pods) != 1:
                continue
            else:
                running_fc_pods = [pod for pod in fc_pods if pod.status.phase == "Running"]
                if len(running_fc_pods) == 1:
                    w.stop()
        print("Frameworkcontroller pod is ready")

    def deploy(self) -> None:
        """
            With the completion of this step the desired architecture of the HPO Framework should be running
            on a platform, e.g,. in the case of Kubernetes it referes to the steps nassary to deploy all pods
            and services in kubernetes.
        """
        # create frameworkcontroller config
        try:
            create_from_yaml(
                self.k8s_api_client,
                path.join(path.dirname(__file__),
                          "nni-template/fc-config.yaml"),
                namespace=self.namespace, verbose=True
            )
        except FailToCreateError as e:
            raise e

        # deploy frameworkcontroller operator
        try:
            create_from_yaml(
                self.k8s_api_client,
                path.join(path.dirname(__file__), "nni-template/fc-sts.yaml"),
                namespace=self.namespace, verbose=True
            )
        except FailToCreateError as e:
            raise e

        self._deploy_watch()

    def setup(self):
        self.experiment.config.nni_manager_ip = self.metricsIP
        self.experiment.config.trial_code_directory = path.join(path.dirname(__file__), 'tune-env')
        self.experiment.config.trial_command = f'export METRICS_STORAGE_HOST="{self.metricsIP}" && python3 tunescript.py'
        self.experiment.config.tuner.name = 'GridSearch'
        self.experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
        self.experiment.config.max_trial_number = n_trials
        self.experiment.config.trial_concurrency = self.workerCount
        self.experiment.config.search_space = global_grid
        self.experiment.config.max_trial_duration = '15m'

        self.experiment.config.training_service.service_account_name = 'frameworkcontroller'
        self.experiment.config.training_service.namespace = self.namespace
        self.experiment.config.training_service.reuse_mode = True

        self.experiment.config.training_service.storage = K8sNfsConfig()
        self.experiment.config.training_service.storage.storage_type = 'nfs'
        self.experiment.config.training_service.storage.server = self.nfsServer
        self.experiment.config.training_service.storage.path = self.nfsPath

        self.experiment.config.training_service.task_roles = [FrameworkControllerRoleConfig()]
        self.experiment.config.training_service.task_roles[0].name = 'worker'
        self.experiment.config.training_service.task_roles[0].task_number = 1
        self.experiment.config.training_service.task_roles[0].docker_image = 'vdocker2603/ml-benchmark-nni-k8s'
        self.experiment.config.training_service.task_roles[0].command = f'export METRICS_STORAGE_HOST="{self.metricsIP}" && python3 tunescript.py'
        self.experiment.config.training_service.task_roles[0].gpu_number = 0
        self.experiment.config.training_service.task_roles[0].cpu_number = self.workerCpu
        self.experiment.config.training_service.task_roles[0].memory_size = f"{self.workerMemory} gb"
        self.experiment.config.training_service.task_roles[0].framework_attempt_completion_policy = FrameworkAttemptCompletionPolicy(
            min_failed_task_count=1, min_succeed_task_count=1)

    def run(self):
        """
            Executing the hyperparameter optimization on the deployed platfrom.
            use the metrics object to collect and store all measurments on the workers.
        """
        self.experiment.run(port=8080, debug=True)
        
        print(self.experiment.list_trial_jobs())
        print(self.experiment.get_job_metrics())

        self.experiment.stop()
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
        return
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

    def _undeploy_watch_fc_operator(self):
        w = watch.Watch()
        for event in w.stream(self.k8s_core_v1_api.list_namespaced_pod, namespace=self.namespace):
            print(
                f"Event: {event['type']} {event['object'].metadata.name} {event['object'].status.phase}")
            resp = self.k8s_core_v1_api.list_namespaced_pod(self.namespace)
            pods = resp.items
            fc_operator_pods = [pod for pod in pods if "frameworkcontroller" in pod.metadata.name]
            if len(fc_operator_pods) != 0:
                continue
            else:
                w.stop()
        print("Frameworkcontroller Operator pod was deleted successfully")

    def undeploy(self):
        """
            The clean-up procedure to undeploy all components of the HPO Framework that were deployed in the
            Deploy step.
        """
        time.sleep(5)
        
        # undeploy fc operator
        try:
            self.k8s_apps_v1_api.delete_namespaced_stateful_set("frameworkcontroller", self.namespace)
        except ApiException as e:
            raise e

        self._undeploy_watch_fc_operator()

        # delete fc config
        try:
            self.k8s_core_v1_api.delete_namespaced_config_map("frameworkcontroller-config", self.namespace)
        except ApiException as e:
            raise e


def create_nni_grid(grid):
    nni_grid = {}
    for key, value in grid.items():
        if type(grid[key]) is list:
            nni_grid[key] = {'_type': 'choice', '_value': value}
        else:
            nni_grid[key] = {'_type': 'choice', '_value': [value]}
    return dict(nni_grid)


if __name__ == "__main__":
    from ml_benchmark.benchmark_runner import BenchmarkRunner
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--grid', help='Grid config, valid option: ["small", "medium", "large"]')
    parser.add_argument(
        '--nworkers', help='Number of workers, valid option: [1, 2, 4]', type=int)

    args = parser.parse_args()
    nworkers = args.nworkers
    grid_option = args.grid
    if nworkers is None:
        print("Number of workers is not specified, default is 1")
        nworkers = 1
    else:
        if nworkers not in [1, 2, 4]:
            raise ValueError(
                "Invalid number of workers: valid option: [1, 2, 4]")

    if grid_option is None:
        print("Grid option is not specified, default is small")
        grid_option = "small"
    else:
        if grid_option not in ["small", "medium", "large"]:
            raise ValueError(
                "Invalid number of workers: valid option: [small, medium, large]")
    n_trials_dict = {
        "small": 8,
        "medium": 16,
        "large": 32
    }
    n_trials = n_trials_dict[grid_option]

    with open("resource_definition.json") as res_def_file:
        resource_definition = json.load(res_def_file)
        resource_definition['workerCount'] = nworkers

    with open(path.join(path.dirname(__file__), "grids", f"grid_{grid_option}.json")) as grid_def_file:
        grid_definition = create_nni_grid(json.load(grid_def_file))
        global_grid = grid_definition

    runner = BenchmarkRunner(
        benchmark_cls=NNIBenchmark, resources=resource_definition)

    runner.run()
