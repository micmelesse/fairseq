alias drun='sudo docker run -it --network=host --runtime=nvidia --ipc=host -v $HOME/dockerx:/dockerx -w /dockerx/fairseq'
drun nvcr.io/nvidia/pytorch:20.02-py3
