import subprocess
from subprocess import check_output
from abc import ABC, abstractmethod
import docker
from basht.config import Path
from docker.tls import TLSConfig
from pathlib import Path as PathlibPath
import os


def builder_from_string(builder_string):
    if builder_string == "docker":
        return DockerImageBuilder
    elif builder_string == "minikube":
        return MinikubeImageBuilder
    else:
        raise ValueError("Unknown builder string: " + builder_string)


class ImageBuilder(ABC):

    @abstractmethod
    def deploy_image(self, image, tag):
        pass

    @abstractmethod
    def cleanup(self, tag):
        pass


class MinikubeImageBuilder(ImageBuilder):

    def __init__(self, minikube_port: int = 2376) -> None:
        self.client = self._setup_docker_client(minikube_port)

    def _setup_docker_client(self, minikube_port: int):
        home_path = PathlibPath.home()
        minikube_ip = subprocess.run(
            ["minikube", "ip"],
            capture_output=True).stdout.decode("utf-8")
        tls_config = TLSConfig(
            ca_cert=os.path.join(home_path, ".minikube/certs/ca.pem"),
            client_cert=(
                os.path.join(home_path, ".minikube/certs/cert.pem"),
                os.path.join(home_path, ".minikube/certs/key.pem")),
            verify=True)
        return docker.DockerClient(f"https://{minikube_ip}:{minikube_port}", use_ssh_client=False, tls=tls_config)

    def deploy_image(self, image, tag, build_context):
        if not isinstance(image, str):
            image = str(image)
        if not isinstance(build_context, str):
            build_context = str(build_context)
        image, logs = self.client.images.build(dockerfile=image, tag=tag, path=build_context)
        [print(line.get("stream")) for line in logs]

    def deploy_image_with_minikube(self, image, tag, build_context):
        call = subprocess.run(
            ["minikube", "image", "build", "-t", tag,  "-f", image, "."], cwd=build_context,
            capture_output=True)  # doesnt seem to work

        if call.returncode != 0:
            print(call.stderr.decode("utf-8").strip("\n"))
            raise Exception("Failed to deploy image")
        print("IMAGE IMAGE ", call.stdout, call.stderr)

        return call.stdout.decode("utf-8").strip("\n")

    def cleanup(self, tag):
        docker.containers.remove(tag)

    def cleanup_minikube(self, tag):
        call = subprocess.run(["minikube", "image", "rm", tag], check=True)
        if call.returncode != 0:
            raise Exception("Failed to cleanup")


class DockerImageBuilder(ImageBuilder):

    def deploy_image(self, image, tag, build_context):
        call = subprocess.run(
            ["docker", "build", "-t", tag,  "-f", image, "."], cwd=build_context,
            check=True)  # doesnt seem to work

        if call.returncode != 0:
            raise Exception("Failed to build image")

        call = subprocess.run(
            ["docker", "push", tag], check=True)  # doesnt seem to work

        if call.returncode != 0:
            print(call.stderr.decode("utf-8").strip("\n"))
            raise Exception("Failed to deploy image")

        return ""

    def cleanup(self, tag):
        call = subprocess.run(
            ["docker", "image", "rm", tag], check=True)  # doesnt seem to work
        if call.returncode != 0:
            raise Exception("Failed to cleanup")


if __name__ == "__main__":
    image_builder = MinikubeImageBuilder()
    image_builder.deploy_image(
        "experiments/optuna_minikube/dockerfile.trial", "test-case", Path.root_path)
