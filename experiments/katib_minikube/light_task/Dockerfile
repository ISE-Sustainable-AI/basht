FROM python:3.9-slim

RUN pip install pip --upgrade

COPY experiments experiments
# COPY data data
COPY setup.py setup.py
COPY ml_benchmark ml_benchmark



RUN pip install -e .

CMD ["python", "experiments/katib_minikube/light_task/task.py"]