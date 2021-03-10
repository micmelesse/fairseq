#!/bin/bash
SCRIPTPATH=$(dirname $(realpath $0))

if [ ! "$BASH_VERSION" ]; then
    echo "Using bash instead of sh to run this script $0" 1>&2
    bash $SCRIPTPATH/remote_download.sh
    exit 1
fi

LOCAL_DUMP=~/Data/fairseq

REMOTE_PATH_LIST=(
    # roberta_base_amd_sgpu_perf_log
    # roberta_base_nvidia_sgpu_perf_log
    # roberta_base_amd_sgpu_timeline
    # roberta_base_amd_sgpu_perf_summary
    # roberta_base_amd_mgpu_perf_scaling
    roberta_base_nvidia_sgpu_perf_timeline
)

for REMOTE_PATH in "${REMOTE_PATH_LIST[@]}"; do
    DOWNLOAD_DIR=${LOCAL_DUMP}/$(dirname $REMOTE_PATH)
    mkdir -p $DOWNLOAD_DIR
    # rsync -av -e "ssh -p 20059" michael@10.216.64.100:~/dockerx/fairseq/$REMOTE_PATH ${DOWNLOAD_DIR}
    rsync -av -e "ssh -p 20187" mmelesse@10.216.64.100:~/dockerx/fairseq/$REMOTE_PATH ${DOWNLOAD_DIR}
    #  rsync -av -e "ssh" mmelesse@fpvega10-3:~/dockerx/fairseq/$REMOTE_PATH ${DOWNLOAD_DIR}
done
