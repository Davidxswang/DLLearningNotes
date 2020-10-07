FROM ubuntu
# update the container's packages
RUN apt update; apt dist-upgrade -y
# install apache2 and vim
RUN apt install -y apache2 vim
# make apache automatically start-up
RUN echo "/etc/init.d/apache2 start" >> /etc/bash.bashrc
# the entrypoint (default command to run)
CMD ["/bin/bash"]