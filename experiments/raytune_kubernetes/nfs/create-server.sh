if [[ -z $(docker images | grep rootsquash-nfs-server) ]]; then
	docker build -t rootsquash-nfs-server ./docker/	
fi

docker run -d --privileged -v $PWD/exports:/exports --expose 2049 --expose 20048 --expose 111 --name=docker-nfs-server --net=host rootsquash-nfs-server

#k8s.gcr.io/volume-nfs 
