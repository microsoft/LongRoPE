#!/bin/bash
cd ../../cube-test/checkpoint
bash gen_test.sh 

# cd ../../script/auto_merge
# python auto_merge.py \
#     --proj-dir /mnt/yiran/cube-new/LongRoPE/cube-new/cube-test \
#     --sh-path /mnt/yiran/cube-new/LongRoPE/cube-new/cube-test/run.sh \
#     --storage "https://fastnn.blob.core.windows.net/teamdrive/ExtendSeqLen/ft_out_model/cube-la2-4k-128k-same-doc/" \
#     --key ""