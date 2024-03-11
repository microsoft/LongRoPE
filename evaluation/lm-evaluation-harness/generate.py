import sys
import json

def print_all(path):
    with open(path, "r") as fp:
        curr = json.load(fp)
    
    all_score = 0
    for task in ["storycloze_2018", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]:
        if task not in curr['results']:
            continue
        if "acc_norm" in curr["results"][task]:
            print(f"{task}: {round(curr['results'][task]['acc']*100, 2)}/{round(curr['results'][task]['acc_norm']*100, 2)}")
        else:
            print(f"{task}: {round(curr['results'][task]['acc']*100, 2)}")
        all_score += curr['results'][task]['acc']
    print("all: ", round(all_score / 7. * 100, 2))
    print("------------------------------")
    
    all_score = 0
    for task in ["boolq", "race_high"]:
        if task not in curr['results']:
            continue
        if "acc_norm" in curr["results"][task]:
            print(f"{task}: {round(curr['results'][task]['acc']*100, 2)}/{round(curr['results'][task]['acc_norm']*100, 2)}")
        else:
            print(f"{task}: {round(curr['results'][task]['acc']*100, 2)}")
        all_score += curr['results'][task]['acc']
    print("all: ", round(all_score / 2. * 100, 2))

print_all(sys.argv[1])
