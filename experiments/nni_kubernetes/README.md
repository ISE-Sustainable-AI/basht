Run NNI Benchmark:
- Step 1: Setup the NFS server if it is not already setup
  - Navigate to the nfs folder
  - Run ```./create-server.sh```
  - Run ```umask 0000``` to suppress errors related to NFS permission
- Step 2: Create a resource_definition.json (e.g. from resource_definition.example.json)
- Step 3: Run the benchmark script ```python3 nni_benchmark.py```
