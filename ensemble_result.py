import pandas as pd
import glob
import numpy as np
import os


class CFG:
    Merge_Top_Solution = True
    result_folder = "./"


def ensemble(data):
    temp = np.zeros_like(data[0].sort_values(by='id')["target"].values)
    for j in range(len(data)):
        temp += 1 / len(data) * data[j].sort_values(by='id')["target"].values
    ensemble_result = pd.DataFrame({
        'id': data[0].sort_values(by='id')["id"].values,
        'target': temp
    })
    return ensemble_result


workdir = r"./model/results/"
if CFG.Merge_Top_Solution:
    files = glob.glob(workdir + r"**/*.csv", recursive=True)
else:
    files = glob.glob(workdir + r"*.csv")
print(len(files), "csv files merged.")
results = []
for file in files:
    results.append(pd.read_csv(file))
save_path = os.path.join(CFG.result_folder, "submission_ensemble.csv")
ensemble(results).to_csv(save_path, index=False)
print("csv write to:", save_path)
