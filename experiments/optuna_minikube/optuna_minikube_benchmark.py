from os import path
from time import sleep
import optuna
from kubernetes import client, config, watch
from kubernetes.client import ApiException
from kubernetes.utils import create_from_yaml, FailToCreateError
from basht.benchmark_runner import Benchmark
from basht.config import Path
from basht.utils.image_build_wrapper import builder_from_string
from basht.workload.objective import Objective
from basht.utils.yaml import YamlTemplateFiller
from basht.utils.yaml import YMLHandler
from basht.utils.generate_grid_search_space import generate_grid_search_space


class OptunaMinikubeBenchmark(Benchmark):
    _path = path.dirname(__file__)

    def __init__(self, resources: dict) -> None:
        """
        Processes the given resources dictionary and creates class variables from it which are used
        in the benchmark.

        Args:
            resources (dict): _description_
        """
        config.load_kube_config(context=resources.get("kubernetesContext"))
        self.namespace = resources.get("kubernetesNamespace")
        # ensures that we can work with kubernetes
        # TODO: if this fails we woun't know about it until we try to run the experiment...
        self.image_builder = builder_from_string(resources.get("dockerImageBuilder"))()
        self.master_ip = resources.get("kubernetesMasterIP")
        self.trial_tag = resources.get("dockerImageTag")
        self.study_name = resources.get("studyName")
        self.workerCpu = resources.get("workerCpu")
        self.workerMemory = resources.get("workerMemory")
        self.workerCount = resources.get("workerCount")
        self.delete_after_run = resources.get("deleteAfterRun")
        self.metrics_ip = resources.get("metricsIP")
        self.trials = resources.get("trials")
        self.search_space = generate_grid_search_space(resources.get("hyperparameter"))
        self.workload = resources.get("workload")

    def deploy(self) -> None:
        """
        Deploy DB
        """

        try:
            resp = client.CoreV1Api().create_namespace(
                client.V1Namespace(metadata=client.V1ObjectMeta(name=self.namespace)))
            print("Namespace created. status='%s'" % str(resp))
        except ApiException as e:
            if self._is_create_conflict(e):
                print("Deployment already exists")
            else:
                raise e
        try:
            resp = create_from_yaml(client.ApiClient(),
                                    path.join(path.dirname(__file__), "ops/manifests/db/db-deployment.yml"),
                                    namespace=self.namespace, verbose=True)
            print("Deployment created. status='%s'" % str(resp))
        except FailToCreateError as e:
            if self._is_create_conflict(e):
                print("Deployment already exists")
            else:
                raise e

        self._watch_db()

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

    def setup(self):
        """

        """
        # TODO: compile grid and task into image
        self.image_builder.deploy_image(
            "experiments/optuna_minikube/dockerfile.trial", self.trial_tag, Path.root_path)
        print(f"Image: {self.trial_tag}")

    def run(self):
        """Start Trial Pods as Jobs.
        """
        job_definition = {
            "worker_num": self.workerCount,
            "worker_cpu": self.workerCpu,
            "worker_mem": f"{self.workerMemory}Gi",
            "worker_image": self.trial_tag,
            "study_name": self.study_name,
            "metrics_ip": self.metrics_ip,
        }
        print(f"Jobs will be created with the following configuration:\n{job_definition}")
        job_yml_objects = YamlTemplateFiller.load_and_fill_yaml_template(
            path.join(path.dirname(__file__), "ops/manifests/trial/job.yml"), job_definition)
        try:
            create_from_yaml(
                client.ApiClient(), yaml_objects=job_yml_objects, namespace=self.namespace, verbose=True)
        except FailToCreateError as e:
            if self._is_create_conflict(e):
                # lets remove the old one and try again
                client.BatchV1Api().delete_namespaced_job(name="optuna-trial", namespace=self.namespace)
                create_from_yaml(
                    client.ApiClient(), yaml_objects=job_yml_objects, namespace=self.namespace, verbose=True)
            else:
                raise e
        self._watch_trials()

    def _getDBURL(self):
        postgres_sepc = client.CoreV1Api().read_namespaced_service(namespace=self.namespace, name="postgres")
        if postgres_sepc:
            if len(postgres_sepc.spec.ports) > 0 and postgres_sepc.spec.ports[0].node_port > 0:
                # XXX: hardcoded credentaials - should match ops/mainifests/db/db-deployment.yaml#ostgres-config
                url = f"postgresql://postgresadmin:admin123@{self.master_ip}:{postgres_sepc.spec.ports[0].node_port}/postgresdb"
                return url
            raise Exception("Postgres DB not found - spec dose not have node port}" + str(postgres_sepc))
        raise Exception("Postgres DB URL not found")

    def collect_run_results(self):
        """Collect results form container or on master?
        """
        study = optuna.load_study(study_name=self.study_name, storage=self._getDBURL())
        self.best_trial = study.best_trial

    def _watch_trials(self):
        """
        Checks if Trials (Kubernetes Jobs) are completed. If not the process waits on it.
        """
        w = watch.Watch()
        c = client.BatchV1Api()
        for e in w.stream(c.list_namespaced_job, namespace=self.namespace, timeout_seconds=10000):
            if "object" in e and e["object"].status.completion_time is not None:
                w.stop()
                return
        sleep(5)
        print("Trials completed! Collecting Results")

    def test(self):

        def optuna_trial(trial):
            if self.search_space.get("hidden_lyer_config"):
                hidden_layer_idx = trial.suggest_categorical(
                    "hidden_layer_config", list(self.search_space["hidden_layer_config"].keys()))
                hidden_layer_config = self.search_space.get("hidden_layer_config")[hidden_layer_idx]
            else:
                hidden_layer_config = None
            if self.search_space.get("learning_rate"):
                lr = trial.suggest_float(
                    "learning_rate", min(self.search_space["learning_rate"]),
                    max(self.search_space["learning_rate"]), log=True)
            else:
                lr = None
            if self.search_space.get("weight_decay"):
                decay = trial.suggest_float(
                    "weight_decay", min(self.search_space["weight_decay"]),
                    max(self.search_space["weight_decay"]), log=True)
            else:
                decay = None
            hyperparameter = {
                "learning_rate": lr, "weight_decay": decay,
                "hidden_layer_config": hidden_layer_config
            }
            hyperparameter = {key: value for key, value in hyperparameter.items() if value}
            objective = Objective(
                dl_framework=self.workload.get("dl_framework"), model_cls=self.workload.get("model_cls"),
                epochs=self.workload.get("epochs"), device=self.workload.get("device"),
                task=self.workload.get("task"), hyperparameter=hyperparameter)
            # these are the results, that can be used for the hyperparameter search
            objective.load()
            objective.train()
            validation_scores = objective.test()
            return validation_scores["macro avg"]["f1-score"]

        self.scores = optuna_trial(self.best_trial)

    def collect_benchmark_metrics(self):
        results = dict(
            test_scores=self.scores
        )
        return results

    def undeploy(self):
        """Kill all containers
        """
        if self.delete_after_run:
            client.CoreV1Api().delete_namespace(self.namespace)
            self._watch_namespace()
            self.image_builder.cleanup(self.trial_tag)

    def _watch_namespace(self):
        c = client.CoreV1Api()
        no_exception = True
        while no_exception:
            try:
                c.read_namespace_status(self.namespace)
                sleep(2)
            except ApiException:
                no_exception = False

    def _watch_db(self):
        w = watch.Watch()
        c = client.AppsV1Api()
        for e in w.stream(
            c.list_namespaced_deployment, namespace=self.namespace, timeout_seconds=10,
                field_selector="metadata.name=postgres"):
            deployment_spec = e["object"]
            if deployment_spec is not None:
                if deployment_spec.status.available_replicas is not None \
                        and deployment_spec.status.available_replicas > 0:
                    w.stop()
                    return True
        return False


if __name__ == "__main__":

    from basht.benchmark_runner import BenchmarkRunner
    import subprocess
    from urllib.request import urlopen
    import os

    # The basic config for the workload. For testing purposes set epochs to one.
    # For benchmarking take the default value of 100
    # your ressources the optimization should run on
    dir_path = os.path.abspath(os.path.dirname(__file__))
    resources = YMLHandler.load_yaml(os.path.join(dir_path, "resource_definition.yml"))
    to_automate = {
        "metricsIP": urlopen("https://checkip.amazonaws.com").read().decode("utf-8").strip(),
        "kubernetesMasterIP": subprocess.check_output("minikube ip", shell=True).decode("utf-8").strip("\n")}
    resources.update(to_automate)

    # import an use the runner
    runner = BenchmarkRunner(
        benchmark_cls=OptunaMinikubeBenchmark, resources=resources)
    runner.run()
