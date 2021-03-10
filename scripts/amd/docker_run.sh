PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
# -u `id -u`:`id -g`
docker run -it --network=host -v=`pwd`:$DOCKER_DIR -w $DOCKER_DIR --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  $1 $2