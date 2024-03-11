import sys
import json

def print_all(path):
    with open(path, "r") as fp:
        curr = json.load(fp)["results"]
    
    res = []
    for item in curr:
        print(item)
        assert curr[item]["acc"] == curr[item]["acc_norm"]
        res.append(curr[item]["acc_norm"])
    print(len(curr))
    print(sum(res)/len(res))

print_all(sys.argv[1])




