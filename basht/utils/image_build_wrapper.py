import subprocess
from abc import ABC, abstractmethod
import docker
from docker.tls import TLSConfig


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

    def __init__(self) -> None:
        pass

    def deploy_image(self, image, tag, build_context):
        call = subprocess.run(
            ["minikube", "image", "build", "-t", tag,  "-f", image, "."], cwd=build_context,
            capture_output=True)  # doesnt seem to work

        if call.returncode != 0:
            print(call.stderr.decode("utf-8").strip("\n"))
            raise Exception("Failed to deploy image")
        print("IMAGE IMAGE ", call.stdout, call.stderr)

        return call.stdout.decode("utf-8").strip("\n")

    def cleanup(self, tag):
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
    # TODO: docker alternative testing
    minikube_ip = subprocess.run(
            ["minikube", "ip"],
            capture_output=True).stdout.decode("utf-8")
    tls_config = TLSConfig(ca_cert="/home/gebauer/.minikube/certs/ca.pem")
    client = docker.DockerClient(f"https://{minikube_ip}:2376", tls=tls_config)
    client.run("hello-world")
