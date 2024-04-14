#!/bin/bash

python auto_merge.py \
    --proj-dir /mnt/yiran/cube-new-cz/cube-la2-32k-512k-12xdoc \
    --sh-path /mnt/yiran/cube-new-cz/cube-la2-32k-512k-12xdoc/run-la2-32k-pose-512k-batch8-12xdoc.sh \
    --storage "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-32k-pose-512k-12xdoc" \
    --key ""