#!/bin/bash
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR_PATH=$(dirname $SCRIPT_PATH)

# clean up
rm -rf checkpoints

# use bash
if [ ! "$BASH_VERSION" ]; then
    echo "Using bash to run this script $0" 1>&2
    exec bash $SCRIPT_PATH "$@"
    exit 1
fi

# check for roberta arcitecure
if [[ "$*" == *"large"* ]]; then
    echo "Roberta Large"
    ROBERTA_ARCH=roberta_large
else
    echo "Roberta Base"
    ROBERTA_ARCH=roberta_base
fi

TRAIN_DIR=$ROBERTA_ARCH

if [[ "$*" == *"nvidia"* ]]; then
    TRAIN_DIR="${TRAIN_DIR}_nvidia"
else
    TRAIN_DIR="${TRAIN_DIR}_amd"
fi

# check if multi gpu
if [[ "$*" == *"mgpu"* ]]; then
    echo "Multi GPU run"
    TRAIN_DIR="${TRAIN_DIR}_mgpu"
elif [[ "$*" == *"scaling"* ]]; then
    echo "Multi GPU scaling test"
    TRAIN_DIR="${TRAIN_DIR}_mgpu"
else
    echo "Single GPU run"
    # choose a single gpu
    if [[ "$*" == *"nvidia"* ]]; then
        export CUDA_VISIBLE_DEVICES=0
    else
        export HIP_VISIBLE_DEVICES=0
    fi

    TRAIN_DIR="${TRAIN_DIR}_sgpu"
fi

# setup run
if [[ "$*" == *"converg"* ]]; then
    echo "Convergence run"
    TOTAL_UPDATES=125000  # Total number of training steps
    WARMUP_UPDATES=10000  # Warmup the learning rate over this many updates
    PEAK_LR=0.0005        # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512 # Max sequence length
    MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=16      # Number of sequences per batch (batch size)
    UPDATE_FREQ=16        # Increase the batch size 16x

    TRAIN_DIR="${TRAIN_DIR}_converg"
elif [[ "$*" == *"debug"* ]]; then
    echo "Debug run"
    TOTAL_UPDATES=10      # Total number of training steps
    WARMUP_UPDATES=1      # Warmup the learning rate over this many updates
    PEAK_LR=0.0005        # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512 # Max sequence length
    MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=8       # Number of sequences per batch (batch size)
    UPDATE_FREQ=16        # Increase the batch size 16x

    TRAIN_DIR="${TRAIN_DIR}_debug"
else
    echo "Performance run"
    TOTAL_UPDATES=1000    # Total number of training steps
    WARMUP_UPDATES=100    # Warmup the learning rate over this many updates
    PEAK_LR=0.0005        # Peak learning rate, adjust as needed
    TOKENS_PER_SAMPLE=512 # Max sequence length
    MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
    MAX_SENTENCES=16      # Number of sequences per batch (batch size)
    UPDATE_FREQ=16        # Increase the batch size 16x

    TRAIN_DIR="${TRAIN_DIR}_perf"
fi

# choose precision
if [[ "$*" == *"fp32"* ]]; then
  echo "Running FP32"
  PRECISION=""
  TRAIN_DIR="${TRAIN_DIR}_fp32"
else
  echo "Running FP16"
  PRECISION="--fp16"
  TRAIN_DIR="${TRAIN_DIR}_fp16"
fi

# setup data dir
DATA_DIR=data/wikitext-103/data-bin/wikitext-103

# train roberta
if [[ "$*" == *"profile"* ]]; then
    TRAIN_DIR="${TRAIN_DIR}_profile"
    rm -rf $TRAIN_DIR
    mkdir -p $TRAIN_DIR

    echo "Profiling run"

    if [[ "$*" == *"nvidia"* ]]; then
        nvprof --normalized-time-unit us --demangling off --log-file $TRAIN_DIR/$TRAIN_DIR.log \
            python3.6 train.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION

        nvprof --normalized-time-unit us --demangling off -o $TRAIN_DIR/$TRAIN_DIR.nvprof \
            python3.6 train.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION

    else

        ROCPROF=$SCRIPT_DIR_PATH/../rocprofiler_pkg_*/rocprofiler_pkg/rocprof

        # run pretraining with rocrpof
        $ROCPROF -i $SCRIPT_DIR_PATH/in.txt --hip-trace --roctx-trace --timestamp on -d rocout \
            python3.6 train.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION

        mv in.db $TRAIN_DIR
        mv in.hip_stats.csv $TRAIN_DIR
        mv in.json $TRAIN_DIR
        mv rocout $TRAIN_DIR
        cp $TRAIN_DIR/rocout/rpl_data_*/input*/log.txt $TRAIN_DIR
    fi

    mv checkpoints $TRAIN_DIR
    chmod -R 777 $TRAIN_DIR
elif [[ "$*" == *"summary"* ]]; then
    TRAIN_DIR="${TRAIN_DIR}_summary"
    rm -rf $TRAIN_DIR
    mkdir -p $TRAIN_DIR

    if [[ "$*" == *"nvidia"* ]]; then
        nvprof --normalized-time-unit us --demangling off --log-file $TRAIN_DIR/$TRAIN_DIR.log \
            python3.6 train.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION
    else

        ROCPROF=$SCRIPT_DIR_PATH/../rocprofiler_pkg_*/rocprofiler_pkg/rocprof

        # run pretraining with rocrpof
        $ROCPROF --timestamp on \
            python3.6 train.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION

        mv results.csv $TRAIN_DIR

        rm -rf checkpoints
        export HCC_PROFILE=2

        python3.6 train.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION \
            2>&1 | tee $TRAIN_DIR/log.txt

        /opt/rocm/hcc/bin/rpt --topn -1 $TRAIN_DIR/log.txt >$TRAIN_DIR/rpt_summary.txt
    fi

    mv checkpoints $TRAIN_DIR
    chmod -R 777 $TRAIN_DIR
elif [[ "$*" == *"timeline"* ]]; then
    TRAIN_DIR="${TRAIN_DIR}_timeline"
    rm -rf $TRAIN_DIR
    mkdir -p $TRAIN_DIR

    # force reduced timesteps
    TOTAL_UPDATES=10 # Total number of training steps
    WARMUP_UPDATES=1 # Warmup the learning rate over this many updates

    if [[ "$*" == *"nvidia"* ]]; then
        echo "Nvidia Timeline"
        nvprof -o $TRAIN_DIR/$TRAIN_DIR.nvprof \
            python3.6 train_profile.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION
    else
        echo "Roctracer Timeline"
        ROCPROF=$SCRIPT_DIR_PATH/../rocprofiler_pkg_*/rocprofiler_pkg/rocprof

        # run pretraining with rocrpof
        # $ROCPROF -i $SCRIPT_DIR_PATH/in.txt --timestamp on -d rocout \
            python3.6 train_profile.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION

        mv in.db $TRAIN_DIR
        mv in.hip_stats.csv $TRAIN_DIR
        mv in.json $TRAIN_DIR
        mv rocout $TRAIN_DIR
        cp $TRAIN_DIR/rocout/rpl_data_*/input*/log.txt $TRAIN_DIR
        cp $TRAIN_DIR/in.json $TRAIN_DIR/timeline.json
    fi

    mv checkpoints $TRAIN_DIR
    chmod -R 777 $TRAIN_DIR
elif [[ "$*" == *"log"* ]]; then

    TRAIN_DIR="${TRAIN_DIR}_log"
    rm -rf $TRAIN_DIR
    mkdir -p $TRAIN_DIR

    echo "Collecting Log at $TRAIN_DIR/log.txt"
    python3.6 train.py $DATA_DIR \
        --task masked_lm --criterion masked_lm \
        --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION \
        2>&1 | tee $TRAIN_DIR/log.txt

    mv checkpoints $TRAIN_DIR
    chmod -R 777 $TRAIN_DIR

elif [[ "$*" == *"scaling"* ]]; then
    pip3 install pandas
    TRAIN_DIR="${TRAIN_DIR}_scaling"
    rm -rf $TRAIN_DIR
    mkdir -p $TRAIN_DIR

    # force reduced timesteps
    TOTAL_UPDATES=10 # Total number of training steps
    WARMUP_UPDATES=1 # Warmup the learning rate over this many updates

    for GPUS in 0 0,1 0,1,2 0,1,2,3 0,1,2,3,4 0,1,2,3,4,5 0,1,2,3,4,5,6 0,1,2,3,4,5,6,7; do
        export HIP_VISIBLE_DEVICES=$GPUS

        CUR_TRAIN_DIR="${TRAIN_DIR}/scaling_$GPUS"
        rm -rf $CUR_TRAIN_DIR
        mkdir -p $CUR_TRAIN_DIR

        python3.6 train.py $DATA_DIR \
            --task masked_lm --criterion masked_lm \
            --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
            --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
            --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
            --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
            --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION \
            2>&1 | tee $CUR_TRAIN_DIR/log.txt

        # mv checkpoints $CUR_TRAIN_DIR
        rm -rf checkpoints
        chmod -R 777 $CUR_TRAIN_DIR
        python3 $SCRIPT_DIR_PATH/get_roberta_metrics.py --roberta_log_path=$CUR_TRAIN_DIR/log.txt
    done
elif [[ "$*" == *"rocblas"* ]]; then
    # force reduced timesteps
    TOTAL_UPDATES=5  # Total number of training steps
    WARMUP_UPDATES=1 # Warmup the learning rate over this many updates
    if [ "$ROBERTA_ARCH" == "roberta_base" ]; then
        MAX_SENTENCES=31
    fi

    if [ "$ROBERTA_ARCH" == "roberta_large" ]; then
        MAX_SENTENCES=15
    fi

    TRAIN_DIR="${TRAIN_DIR}_rocblas_batch${MAX_SENTENCES}"
    rm -rf $TRAIN_DIR
    mkdir -p $TRAIN_DIR

    echo "ROCBLAS Trace"
    export ROCBLAS_LAYER=3
    export ROCBLAS_LOG_TRACE_PATH=$TRAIN_DIR/rocblas_log_trace.txt
    export ROCBLAS_LOG_BENCH_PATH=$TRAIN_DIR/rocblas_log_bench.txt

    python3.6 train.py $DATA_DIR \
        --task masked_lm --criterion masked_lm \
        --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION

    mv checkpoints $TRAIN_DIR
    chmod -R 777 $TRAIN_DIR

else
    rm -rf $TRAIN_DIR
    mkdir -p $TRAIN_DIR

    python3.6 train.py $DATA_DIR \
        --task masked_lm --criterion masked_lm \
        --arch $ROBERTA_ARCH --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 $PRECISION

    mv checkpoints $TRAIN_DIR
    chmod -R 777 $TRAIN_DIR
fi
