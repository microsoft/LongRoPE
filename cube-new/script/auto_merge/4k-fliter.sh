#!/bin/bash

python auto_merge.py \
    --proj-dir /mnt/yiran/cube-new/cube-la2-4k-128k-same-doc \
    --sh-path /mnt/yiran/cube-new/cube-la2-4k-128k-same-doc/run-la2-4k-pose-128k-batch8.sh \
    --storage "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-4k-128k-same-doc/" \
    --key "?sv=2023-01-03&st=2024-04-22T12%3A15%3A28Z&se=2024-04-25T12%3A15%3A00Z&sr=c&sp=racwdl&sig=lQc6Vz%2BpSby%2BbHMe%2B64MLO1hXEKiroRXAio7DhvIvSE%3D"