FROM python:3.8-slim
RUN pip install nni
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

COPY setup.py setup.py
COPY ml_benchmark ml_benchmark
RUN pip install -e .

CMD ["/bin/bash"]