import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_result_distribution(path):
    data = pd.read_csv(path)
    sns.distplot(data["target"])
    plt.show()


plot_result_distribution("./submission_ensemble.csv")
