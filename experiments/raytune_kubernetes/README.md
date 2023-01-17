Run Raytune Benchmark:
- Step 1: build a docker container `docker build -t raytune_kubernetes -f experiments/raytune_kubernetes/docker/Dockerfile .`
- Step 2: Create a resource_definition.json (e.g. from resource_definition.example.json)
- Step 3: Run the benchmark script ```python3 raytune_benchmark.py```


# Additional:
## Minikube
- I took out the service account name in the ray-operator.yml
- I set image pull policy in the operator yml to ifnotpresent
- I have to find out the current image version that is used
- I added service account creation
- I added role, rolebindings and custom resource creation
- to delete raycluster edit the raycluster and delete the finalizer: https://stackoverflow.com/questions/71164028/disabling-ray-finalizer-condition or 'kubectl patch rayclusters.cluster.ray.io ray-cluster -p '{"metadata":{"finalizers":null}}' --type=merge'
- i changed the objective function into a static method of the class, to avoid pickling errors
- i changed the grid creation for raytune, to avoid errors with numpy and whatsoever
- "tawalaya/raytune_trial_image:latest" needs to be prebuild and pushed to a public repo

## Kubernetes
- for zfs storage class, i had to create a change the security context of the pod to run as root otherwise we had permission issues