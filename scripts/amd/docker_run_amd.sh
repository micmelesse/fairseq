alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

VOLUMES="-v $HOME/dockerx:/dockerx"
WORK_DIR='-w /root/fairseq'

# IMAGE_NAME=rocm/pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
IMAGE_NAME=fairseq_ibm
CONTAINER_NAME=${IMAGE_NAME}_container

# cd repo && git clone https://github.com/huggingface/transformers && cd ..

drun -d --name $CONTAINER_NAME $WORK_DIR $VOLUMES $IMAGE_NAME
docker cp scripts/amd $CONTAINER_NAME:/root/fairseq/scripts
docker attach $CONTAINER_NAME
