# Select base Image
FROM rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch

# Install dependencies
RUN apt update && apt install -y \
    unzip 

# Download data
RUN mkdir /data  && \
    cd /data &&\
    wget https://raw.githubusercontent.com/nyu-mll/jiant-v1-legacy/master/scripts/download_glue_data.py && \
    python download_glue_data.py --data_dir glue_data --tasks all 

# install fairseq
RUN cd /root && \
    git clone https://github.com/pytorch/fairseq  &&\
    cd fairseq  && \
    pip install --editable ./

# preprocess GLUE data for roberta
RUN cd /root/fairseq && \
    ./examples/roberta/preprocess_GLUE_tasks.sh /data/glue_data ALL

# set work dir
WORKDIR /root/fairseq


