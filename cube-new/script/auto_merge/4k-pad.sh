#!/bin/bash

python auto_merge.py \
    --proj-dir /mnt/yiran/cube-new/cube-la2-4k-128k-same-doc-pad \
    --sh-path /mnt/yiran/cube-new/cube-la2-4k-128k-same-doc-pad/run-la2-4k-pose-128k-batch8-pad.sh \
    --storage "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-4k-128k-same-doc-pad/" \
    --key ""