from pathlib import Path
from os.path import exists

__version__ = "develop"
install_requires = [
        "scikit-learn==0.24.2",
        "tqdm==4.62.3", "SQLAlchemy==1.4.31", "docker==4.4.2",
        "psycopg2-binary",
        "prometheus-api-client==0.5.1",
        "ruamel.yaml==0.17.21"],
test_install_requires = ["pytest==7.1.2", "pytest-cov==3.0.0"]
URL = "https://github.com/gebauerm/ml_benchmark"

long_description = ""

this_directory = Path(__file__).parent.parent
if exists(this_directory/"README.md"):
    long_description = (this_directory / "README.md").read_text()
else:
    long_description = ""
