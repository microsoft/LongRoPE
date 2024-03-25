
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
import argparse

FOLDER_PATH = "./evaluation/needle/"
# LOG_PATH = "longlora-100k"
# LOG_PATH = "longrope_128k_static"
# LOG_PATH = "longrope_256k"
# LOG_PATH = "longrope_128k"
LOG_PATH = "longrope_mis_128k"
# LOG_PATH = "la2"

MODEL_NAME = LOG_PATH
PRETRAINED_LEN=4096

def main(args):
    # Path to the directory containing JSON results
    LOG_PATH = args.path
    MODEL_NAME = args.name
    # /mnt/yiran/LongRoPE/evaluation/needle/result/longrope_mis_128k_debug_ant_1/ck-400
    folder_path = args.path
    print("folder_path", folder_path)
    # MODEL_NAME = LOG_PATH
    # if("/" in folder_path):
    #     model_name = folder_path.split("/")[-2]
    # else: 
    model_name = MODEL_NAME

    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    # import ipdb; ipdb.set_trace()
    # print(json_files)
    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            if not "origin" in args.name:
                score = len(set(model_response.split()).intersection(set(expected_answer))) / len(expected_answer)
            else:
                score = int(json_data.get("score", None)) / 10
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })
            print("context_length", context_length)

    # Creating a DataFrame
    df = pd.DataFrame(data)
    # print(df)
    # print(df["Context Length"])
    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): break
    pretrained_len = li

    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )


    # More aesthetics
    model_name_ = MODEL_NAME
    plt.title(f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    save_path = f"{FOLDER_PATH}img/%s.png" % model_name
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    main(args)