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
- I set image pull policy in the operator yml to ifnotpresent
- I have to find out the current image version that is used
- I added service account creation
- I added role, rolebindings and custom resource creation
- to delete raycluster edit the raycluster and delete the finalizer: https://stackoverflow.com/questions/71164028/disabling-ray-finalizer-condition or 'kubectl patch rayclusters.cluster.ray.io ray-cluster -p '{"metadata":{"finalizers":null}}' --type=merge'
- create exports/raytune folder, for mounting the nfs docker container
- changed the image in the cluster template to the one that is build locally and pushed to the public docker registry
- i changed the objective function into a static method of the class, to avoid pickling errors
- i changed the grid creation for raytune, to avoid errors with numpy and whatsoever


## TODO:

- nodeport statt proxy
- nfs in cluster
- metrics recording

