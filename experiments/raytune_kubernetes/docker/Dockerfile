FROM rayproject/ray:2.1.0-py39
WORKDIR /home/ray/

RUN conda install pytorch==1.11.0 torchvision==0.12.0 cpuonly -c pytorch
COPY setup.py setup.py
COPY basht basht
COPY experiments/raytune_kubernetes experiments/raytune_kubernetes
RUN pip install -e .
RUN mkdir ray-results

CMD ["bash"]
