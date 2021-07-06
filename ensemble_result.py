import pandas as pd
import glob


def ensemble(data):
    temp = data[0].copy()
    for j in range(len(data) - 1):
        temp["target"] += data[j + 1]["target"]
    temp["target"] = 1 / len(data) * temp["target"]
    return temp


workdir = r"./model/results/"
files = glob.glob(workdir + r"*.csv")
results = []
for file in files:
    results.append(pd.read_csv(file))
ensemble(results).to_csv("submission_ensemble.csv", index=False)
