# Use the bitnami Spark image as it comes pre-configured with necessary Spark components
FROM bitnami/spark:latest

USER root

#RUN apt-get update && apt-get install -y openssh-server
#
#RUN mkdir /var/run/sshd
#
#RUN echo "root:password" | chpasswd
#
#RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

# Add a user to run the application
RUN useradd -ms /bin/bash spark
# Create directories for data and jobs
RUN mkdir -p /data/inputs /data/outputs /jobs
# Set the ownership of the directories to the spark user
RUN chown -R spark:spark /opt/bitnami/spark /data /jobs

# Expose SSH port
#EXPOSE 22

# Start SSH service
#CMD ["/usr/sbin/sshd", "-D"]