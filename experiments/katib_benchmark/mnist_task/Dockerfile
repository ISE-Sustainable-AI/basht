FROM python:3.9-slim

ADD . /opt/mnist_task
WORKDIR /opt/mnist_task

# Add folder for the logs.
RUN mkdir /katib
RUN pip install --no-cache-dir -r requirements.txt

RUN chgrp -R 0 /opt/mnist_task \
  && chmod -R g+rwX /opt/mnist_task \
  && chgrp -R 0 /katib \
  && chmod -R g+rwX /katib

ENTRYPOINT ["python3", "/opt/mnist_task/task.py"]
