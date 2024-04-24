rm checkpoint*.pt
rm -rf ck-*

for id in {0..3}; do
    touch checkpoint_1_200-shard${id}.pt
    touch checkpoint_2_300-shard${id}.pt
    touch checkpoint_4_500-shard${id}.pt
    touch checkpoint1-shard${id}.pt
done

mv checkpoint_4_500-shard3.pt checkpoint_4_500-shard3.pt.tmp

