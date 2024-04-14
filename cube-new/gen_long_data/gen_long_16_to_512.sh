
#!/bin/bash

mkdir -p /mnt/yiran/cache/Long-Data-Collections-mistral-512k
python gen_long_data_muti.py \
    --source_data /mnt/yiran/cache/Long-Data-Collections-mistral-16k \
    --mapped_save_path /mnt/yiran/cache/Long-Data-Collections-mistral-512k \
    --scale 32