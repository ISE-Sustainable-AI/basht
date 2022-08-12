FROM rayproject/ray:latest-py38
WORKDIR /home/ray/

RUN conda install pytorch==1.11.0 torchvision==0.12.0 cpuonly -c pytorch
COPY setup.py setup.py
COPY ml_benchmark ml_benchmark
RUN pip install -e .
RUN mkdir ray-results

CMD ["bash"]