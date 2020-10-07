# Docker

To install Docker on Ubuntu 20.04, refer to the [install script](docker-ce-install.sh).

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

## docker rmi <image_id or repository:tag>

Example: `docker rmi ubuntu:latest` or `docker image rmi 4e2eef94cd6b`

This will:

- remove the image from the local machine

## docker rm <container_id>

Example: `docker rm 977eb67441ff`

This will:

- remove the container from the local machine

## docker run [options] \<image-name> [command] [args]

Example: `docker run hello-world`

This will:

- run the command in a new container based on the image specified.
- the same with `docker container run`
- will start an interactive bash session if `-it` is specified, e.g., `docker run -it ubuntu /bin/bash`
- will give the container a name if `--name` is specified, e.g., `docker run --name testubuntu ubuntu:latest`

## docker start <container-id or container-name>

Example: `docker start testubuntu`

This will:

- start a container specified by container id or container name

## docker exec <container-id or container-name> \<command> [args]

Example: `docker exec testubuntu echo "abcd"`

This will:

- run the command in the specified running container

## docker stop <container-id or container-name>

Example: `docker stop testubuntu`

This will:

- stop the specified container

