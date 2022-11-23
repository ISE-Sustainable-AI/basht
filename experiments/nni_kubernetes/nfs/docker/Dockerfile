FROM centos:latest

RUN umask 0000

# Install nfs-utils
RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

RUN yum -y install /usr/bin/ps nfs-utils && yum clean all
RUN yum -y install net-tools
RUN yum -y install vim

# Copy entrypoint
ADD run_nfs.sh /usr/local/bin/run_nfs.sh

# Create exports dir
RUN mkdir -p /exports	\
 && chmod +x /usr/local/bin/run_nfs.sh

# Export NFS Ports
EXPOSE 20048/tcp 2049/tcp 111/tcp

# Expose volume
VOLUME /exports

# Launch entrypoint
ENTRYPOINT ["/usr/local/bin/run_nfs.sh"]

CMD ["/exports"]


