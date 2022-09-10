import datetime
import logging
from threading import Timer

from prometheus_api_client import PrometheusConnect

from ml_benchmark.metrics import NodeUsage
from ml_benchmark.metrics_storage import MetricsStorageStrategy


class RepeatTimer(Timer):

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class ResourceTracker:

    # update every 2 seconds ... maybe make this tuneable
    UPDATE_INTERVAL = 2

    def __init__(self, prometheus_url, resouce_store=MetricsStorageStrategy ):
        if prometheus_url is None:
            raise ValueError("Prometheus URL is required.")
        self.prometheus_url = prometheus_url
        self.prm = PrometheusConnect(url=self.prometheus_url, disable_ssl=True)

        if not self.prm.check_prometheus_connection():
            raise ValueError("Could not connect to Prometheus.")

        self.store = resouce_store()
        self.store.setup()

        self.timer = RepeatTimer(self.UPDATE_INTERVAL, self.update)

        self._check_metrics()

        self.namespace = None

    def _check_metrics(self):
        available = set(self.prm.all_metrics())

        #check node_exporter metrics - cpu/memory
        required = {"node_memory_MemFree_bytes", "node_memory_MemTotal_bytes", "node_cpu_seconds_total","scaph_host_power_microwatts","scaph_process_power_consumption_microwatts"}
        if not required.issubset(available):
            raise ValueError("Prometheus does not provide the required metrics.")

        #check if prometheus is managing a kubernetes cluster
        if "container_network_transmit_bytes_total" in available:
            self.network_metric = "container_network"
        elif "node_network_transmit_bytes_total" in available:
            self.network_metric = "node_network"
        else:
            raise ValueError("Prometheus does not provide a vaild network metric.")

        if "kube_node_info" in available:
            info = self.prm.get_current_metric_value("kube_node_info")
            self.node_map = dict(map(lambda x: (x["internal_ip"], x["node"]), map(lambda x: x["metric"], info)))
        else:
            self.node_map = {}
        

    def update(self):
        try:
            self.track()
        except Exception as e:
            logging.exception("Error while updating resource tracker. %s", e)

    def _query(self):
        """
        Query Prometheus for the current resource usage.
        """
        # ? is there a better way to map nodes using the node_exporter
        memory = 'avg by (instance) (node_memory_MemFree_bytes/node_memory_MemTotal_bytes)'
        cpu = '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[2m])*100))'
        
        ##needs mapping
        network = f'sum by (instance) (rate({self.network_metric}_receive_bytes_total[2m])+rate({self.network_metric}_transmit_bytes_total[2m]))'
        #TODO: reduce measurments to only the ones we care about - dose currently not work with scaph_process_power_consumption_microwatts
        #if we can we collect the power consumption from the scaph_host_power_microwatts metric only for the used namespace
        # if self.namespace:
        #     wattage = f'sum by (node) (scaph_process_power_consumption_microwatts{{namespace="{self.namespace}"}})'
        #     processes = f'count by (node) (scaph_process_power_consumption_microwatts{{namespace="{self.namespace}"}})'
        # else :
        wattage = f'sum by (node) (scaph_host_power_microwatts)'
        processes = 'count by (node) (scaph_process_power_consumption_microwatts)'

        mem_result = self.prm.custom_query(memory)
        cpu_result = self.prm.custom_query(cpu)
        network_result = self.prm.custom_query(network)
        wattage_result = self.prm.custom_query(wattage)
        processes_result = self.prm.custom_query(processes)

        logging.debug("Got results from Prometheus.", mem_result, cpu_result, network_result)

        # assert len(mem_result) == len(cpu_result) == len(network_result)

        #grab the data per instance
        mem_result = dict(map(lambda x: (self._try_norm(x["metric"]["instance"]), float(x["value"][1])), mem_result))
        cpu_result = dict(map(lambda x: (self._try_norm(x["metric"]["instance"]), float(x["value"][1])), cpu_result))
        network_result = dict(map(lambda x: (self._try_norm(x["metric"]["instance"]), float(x["value"][1])), network_result))
        wattage_result = dict(map(lambda x: (self._try_norm(x["metric"]["node"]), float(x["value"][1])), wattage_result))
        processes_result = dict(map(lambda x: (self._try_norm(x["metric"]["node"]), float(x["value"][1])), processes_result))

        logging.debug("Processed Prometheus Results", mem_result, cpu_result, network_result, wattage_result, processes_result)

        # assert mem_result.keys() == cpu_result.keys() == network_result.keys()

        #merge the data
        data = []
        for instance in mem_result:
            n = NodeUsage(instance)
            n.timestamp = datetime.datetime.now()
            n.cpu_usage = cpu_result.get(instance, 0)
            n.memory_usage = mem_result.get(instance, 0)
            n.network_usage = network_result.get(instance, 0)
            if instance in wattage_result:
                n.wattage = wattage_result[instance]
                n.processes = processes_result[instance]
            else:
                n.wattage = -1
                n.processes = -1
            
            data.append(n)
            # logging.debug("Added node usage for %s", instance)
        
        return data

    def track(self):
        data = self._query()

        #insert the data
        for n in data:
            self.store.store(n,table_name="resources")

    def _try_norm(self, instance: str):
        if instance in self.node_map:
            return self.node_map[instance]
        elif instance[:instance.find(":")] in self.node_map:
            return self.node_map[instance[:instance.find(":")]]
        else:
            return instance

    def start(self):
        logging.debug("Starting resource tracker.")
        self.timer.start()

    def stop(self):
        logging.debug("Stopping resource tracker.")
        self.timer.cancel()
