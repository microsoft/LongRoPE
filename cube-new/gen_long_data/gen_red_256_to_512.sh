
#!/bin/bash

mkdir -p /mnt/yiran/cache/RedPajama-512k
python gen_long_data_muti.py \
    --source_data /mnt/yiran/cache/RedPajama-128k \
    --mapped_save_path /mnt/yiran/cache/RedPajama-512k \
    --scale 4