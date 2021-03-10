# FILE_PATH=$1
# FILE=`basename "$FILE_PATH"`
# FILENAME="${FILE%.*}"
# docker build --network=host -t $FILENAME -< $FILE_PATH
docker build --network=host -t rocm/pytorch-private:rocm4.0_ubuntu18.04_py3.6_pytorch_fairseq .