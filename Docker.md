# Docker

To install Docker on Ubuntu 20.04, refer to the [install script](docker-ce-install.sh).

## docker info

Example: `docker info`

This will:

- print very useful information about the docker environment

## docker search \<some-image>

Example: `docker search ubuntu`

This will:

- search the image on docker hub

## docker pull \<image-name>

Example: `docker pull ubuntu`

This will:

- download an image from docker hub

## docker images

Example: `docker images`

This will:

- show what images are available on the local machine

## docker container [command] [options]

Example: `docker container ls -a`

This will:

- print out all the containers including the stopped ones
- print out only the running containers if `-a` is not specified

## docker rmi \<image_id or repository:tag>

Example: `docker rmi ubuntu:latest` or `docker image rmi 4e2eef94cd6b`

This will:

- remove the image from the local machine

## docker rm \<container_id>

Example: `docker rm 977eb67441ff`

This will:

- remove the container from the local machine

## docker run [options] \<image-name> [command] [args]

Example: `docker run hello-world`

This will:

- run the command in a new container based on the image specified.
- the same with `docker container run`
- will start an interactive (-i) psuedo-tty bash (-t) session if `-it` is specified, e.g., `docker run -it ubuntu /bin/bash`
- will give the container a name if `--name` is specified, e.g., `docker run --name testubuntu ubuntu:latest`
- keep the container running in the background (detached mode) even it's not running any command, using `-d`, e.g., `docker run -dit ubuntu /bin/bash`
- redirect a port from the host to the container using `-p`, e.g., `docker run -it -p 8080:80 ubuntu /bin/bash`, which will redirect the 8080 port of the host to the 80 port of the container

## docker start \<container-id or container-name>

Example: `docker start testubuntu`

This will:

- start a container specified by container id or container name, which has already been created based on an image

## docker exec \<container-id or container-name> \<command> [args]

Example: `docker exec testubuntu echo "abcd"`

This will:

- run the command in the specified running container

## docker stop \<container-id or container-name>

Example: `docker stop testubuntu`

This will:

- stop the specified container

## docker ps

Example: `docker ps`

This will:

- list all the running containers
- list all the containers including the stopped ones if `-a` is specified

## docker attach \<container-id or container-name>

Example: `docker attach testubuntu`

This will:

- attach the standard input, output and error streams to the specified running container (as a root user)

## Ctrl [p -> q]

Meaning: press control and p, then don't let go the control key, press the q key

This will:

- exit a container but don't stop the container
- be very useful if we use `docker attach` to enter a container and want to exit it but don't want to stop it

## docker commit [options] \<container-id or name> [repository:[tag]]

Example: `docker commit apacheubuntu ubuntu/apache-server:1.0`

This will:

- create an image based on the container specified

## Dockerfile

Dockerfile is a script to create a docker image automatically.

Example:

```Dockerfile
FROM ubuntu
# update the container's packages
RUN apt update; apt dist-upgrade -y
# install apache2 and vim
RUN apt install -y apache2 vim
# make apache automatically start-up
RUN echo "/etc/init.d/apache2 start" >> /etc/bash.bashrc
# the entrypoint (default command to run)
CMD ["/bin/bash"]
```

This will:

- use a base image specified by `FROM`
- run the command in the image being created specified by `RUN`
- specify a default command for the container to run specified by `CMD`

## docker build -t \<image-name:tag> .

Example: `docker build -t ubuntu/apache-server:1.0 .`

This will:

- use the Dockerfile in the current directory to create an image, so the `.` at the end of the command is important