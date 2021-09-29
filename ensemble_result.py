import pandas as pd
import glob
import numpy as np
import os


class CFG:
    Merge_Top_Solution = False
    result_folder = "./model/results"
    file_name = "submission_final"
    weights = [0.5, 0.5]
    use_weights = False
    acc = [0.8726, 0.8711]
    acc_weights = acc / np.sum(acc)
    use_acc_weights = False


def ensemble(data):
    temp = np.zeros_like(data[0].sort_values(by='id')["target"].values)
    for j in range(len(data)):
        if CFG.use_weights:
            temp += CFG.weights[j] * data[j].sort_values(by='id')["target"].values
        elif CFG.use_acc_weights:
            temp += CFG.acc_weights[j] * data[j].sort_values(by='id')["target"].values
        else:
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
    print(file)
    results.append(pd.read_csv(file))
save_path = os.path.join(CFG.result_folder, CFG.file_name + ".csv")
ensemble(results).to_csv(save_path, index=False)
print("csv write to:", save_path)
