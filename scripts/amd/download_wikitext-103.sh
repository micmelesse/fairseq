# setup directory
DATA_DIR=data/wikitext-103
rm -rf $DATA_DIR
mkdir -p $DATA_DIR

# download and extract data
wget -O $DATA_DIR/wikitext-103-raw-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip $DATA_DIR/wikitext-103-raw-v1.zip -d $DATA_DIR

# encode data to bpe
mkdir -p $DATA_DIR/gpt2_bpe
wget -O $DATA_DIR/gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O $DATA_DIR/gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
for SPLIT in train valid test; do
    python3 -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json $DATA_DIR/gpt2_bpe/encoder.json \
        --vocab-bpe $DATA_DIR/gpt2_bpe/vocab.bpe \
        --inputs $DATA_DIR/wikitext-103-raw/wiki.${SPLIT}.raw \
        --outputs $DATA_DIR/wikitext-103-raw/wiki.${SPLIT}.bpe \
        --keep-empty \
        --workers 60
done

# preprocess and split encoded data
wget -O $DATA_DIR/gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
python3 fairseq_cli/preprocess.py \
    --only-source \
    --srcdict $DATA_DIR/gpt2_bpe/dict.txt \
    --trainpref $DATA_DIR/wikitext-103-raw/wiki.train.bpe \
    --validpref $DATA_DIR/wikitext-103-raw/wiki.valid.bpe \
    --testpref $DATA_DIR/wikitext-103-raw/wiki.test.bpe \
    --destdir $DATA_DIR/data-bin/wikitext-103 \
    --workers 60
