Run Raytune Benchmark:
- Step 1: Setup the NFS server if it is not already setup
  - Navigate to the nfs folder
  - Run ```./create-server.sh```
- Step 2: Create a resource_definition.json (e.g. from resource_definition.example.json)
- Step 3: Run the benchmark script ```python3 raytune_benchmark.py```


# Additional:
## Minikube
- I started nfs on minikube
- I took out the service account name in the ray-operator.yml
- I set image pull policy in the operator yml to always
- I changed to a specific image version
