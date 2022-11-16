from basht.utils.image_build_wrapper import MinikubeImageBuilder
from basht.config import Path
import subprocess
import pytest
import os


def test_minikube_image_builderr_docker():
    # setup
    subprocess_response = subprocess.run(["minikube", "ip"], capture_output=True)
    if subprocess_response.returncode != 0:
        pytest.skip("Minikube is not running, therefore test is skipped.")
    image_builder = MinikubeImageBuilder()
    image_tag = "test_case"
    image_path = os.path.join(Path.test_path, "dockerfile.test")

    # perform
    image_builder.deploy_image(
        image_path, image_tag, str(Path.root_path))

    # assert
    assert image_builder.client.images.get(image_tag)

    # cleanup
    image_builder.cleanup(image_tag)
