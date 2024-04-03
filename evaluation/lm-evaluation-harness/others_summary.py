import os
import sys
import json
import mlflow


mlflow.autolog()


def update_summary(output_dir, item, res):
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as fp:
            summary = json.loads(fp.read())
    else:
        summary = {}
    summary[item] = res
    with open(summary_path, "w") as fp:
        fp.write(json.dumps(summary, indent=4))


def print_all(path):
    with open(path, "r") as fp:
        curr = json.load(fp)["results"]

    for item in curr:
        if "acc_norm" in curr[item]:
            res = curr[item]["acc_norm"]
        else:
            res = curr[item]["mc2"]

        res = round(res * 100, 2)
        print(item, res)
        update_summary(os.path.split(path)[0], item, res)
        try:
            mlflow.log_metric(path, res)
        except:
            pass


print_all(sys.argv[1])
