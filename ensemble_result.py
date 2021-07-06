import pandas as pd
import glob


def ensemble(DataFrame):
    temp = DataFrame[0].copy()
    for j in range(len(DataFrame) - 1):
        temp["target"] += DataFrame[j + 1]["target"]
    temp["target"] = 1 / len(DataFrame) * temp["target"]
    return temp


workdir = r"./model/results/"
files = glob.glob(workdir + r"*.csv")
DataFrame = []
for file in files:
    DataFrame.append(pd.read_csv(file))
ensemble(DataFrame).to_csv("submission_ensemble.csv", index=False)
