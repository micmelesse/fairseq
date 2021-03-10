alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx'
# WORK_DIR=/var/lib/jenkins/pytorch
WORK_DIR='/dockerx/fairseq'


# drun rocm/pytorch:rocm2.10_ubuntu18.04_py3.6_pytorch_profiling
# drun rocm/pytorch-private:rocm3.1_rc2_ubuntu16.04_py3.6_pytorch
# drun -w $WORK_DIR rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
drun rocm/pytorch-private:rocm4.0_ubuntu18.04_py3.6_pytorch_fairseq