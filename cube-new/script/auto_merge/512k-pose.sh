#!/bin/bash

python auto_merge.py \
    --proj-dir /mnt/yiran/cube-new-cz/cube-la2-128k-pose-512k-splice \
    --sh-path /mnt/yiran/cube-new-cz/cube-la2-128k-pose-512k-splice/run-la2-128k-pose-512k-batch8.sh \
    --storage "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-128k-pose-512k-splice/" \
    --key "?sv=2023-01-03&st=2024-04-22T12%3A15%3A28Z&se=2024-04-25T12%3A15%3A00Z&sr=c&sp=racwdl&sig=lQc6Vz%2BpSby%2BbHMe%2B64MLO1hXEKiroRXAio7DhvIvSE%3D"