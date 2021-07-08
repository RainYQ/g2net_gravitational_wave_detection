import pandas as pd
import glob


class CFG:
    Merge_Top_Solution = False


def ensemble(data):
    temp = data[0].copy()
    for j in range(len(data) - 1):
        temp["target"] += data[j + 1]["target"]
    temp["target"] = 1 / len(data) * temp["target"]
    return temp


workdir = r"./model/results/"
if CFG.Merge_Top_Solution:
    files = glob.glob(workdir + r"**/*.csv", recursive=True)
else:
    files = glob.glob(workdir + r"*.csv")
print(len(files), "csv files merged.")
results = []
for file in files:
    results.append(pd.read_csv(file))
ensemble(results).to_csv("submission_ensemble.csv", index=False)
