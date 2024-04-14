python single_test_code.py

ngpus=4
torchrun --nproc_per_node $ngpus test_code.py