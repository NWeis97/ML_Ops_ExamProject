import matplotlib.pyplot as plt
import pandas as pd

# Graphics
import seaborn as sns

sns.set_style("whitegrid")
sns.set_theme()
# Debuging
# import pdb


class Visualizer(object):
    # Folderpath to images
    fpath = "reports/figures/features/"

    # Load data
    train_set = pd.read_csv("./data/processed/train_processed.csv", sep=",")

    # Plot histogram of training data targets #########
    plt.bar(
        ["Disaster", "NotDisaster"], [(train_set.target == 1).sum(), (train_set.target == 0).sum()]
    )
    plt.title("Target distribution of training data")
    plt.savefig("reports/figures/targetdist_training.png", dpi=300)


if __name__ == "__main__":
    Visualizer()
