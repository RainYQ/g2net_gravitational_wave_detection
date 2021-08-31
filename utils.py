import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_result_distribution(path):
    data = pd.read_csv(path)
    plt.figure()
    sns.histplot(data["target"], kde=True, stat='density')
    plt.show()


plot_result_distribution("./submission_ensemble.csv")
