#!/bin/bash
ck_step=ck-1_500
# for city_idx in $(seq 42 3 63); do  
#     echo "city_idx: $city_idx"
#     name=24-cube_longrope_mis_256k_bf16_from500_ck_1_500_debug_needle_origin_city_$city_idx
#     python evaluation/needle/visualize.py --name 24-city-bf16-256k-origin-${city_idx} --path evaluation/needle/result/$name/$ck_step/

# done

# /mnt/yiran/LongRoPE/evaluation/needle/result/25-cube_longrope_mis_256k_bf16_from500_ck_1_500_debug_needle_origin_rnd_5061234/ck-1_500/ck-1_500_len_400000_depth_0_results.json

# rnd_list=(1023456 2034561 3045612 4056123 5061234 6072345 7083456 
rnd_list=(8094567 9105678 1206789 8794563 4352619 6914325 3106748 5623194 7483291 1236087 9078561 2541893 6309572)
for rand_idx in "${rnd_list[@]}"; do  
    echo "rand_idx: $rand_idx"
    name=25-cube_longrope_mis_256k_bf16_from500_ck_1_500_debug_needle_origin_rnd_$rand_idx
    python evaluation/needle/visualize.py --name 25-nums-bf16-256k-origin-${rand_idx} --path evaluation/needle/result/$name/$ck_step/
done