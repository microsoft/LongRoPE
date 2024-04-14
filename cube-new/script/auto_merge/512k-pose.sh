#!/bin/bash

python auto_merge.py \
    --proj-dir /mnt/yiran/cube-new-cz/cube-la2-128k-pose-512k-splice \
    --sh-path /mnt/yiran/cube-new-cz/cube-la2-128k-pose-512k-splice/run-la2-128k-pose-512k-batch8.sh \
    --storage "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-128k-pose-512k-splice/" \
    --key ""