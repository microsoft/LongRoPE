#!/bin/bash

python auto_merge.py \
    --proj-dir /mnt/yiran/cube-new/cube-la2-4k-128k-same-doc-2xdoc \
    --sh-path /mnt/yiran/cube-new/cube-la2-4k-128k-same-doc-2xdoc/run-la2-4k-pose-128k-batch8-fliter-2xdoc.sh \
    --storage "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-4k-128k-same-doc-fliter-2xdoc" \
    --key ""