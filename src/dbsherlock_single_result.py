import csv
import pickle

import numpy as np
from matplotlib import pyplot as plt

from src.utils import calculate_mean_conf


def calculate_moc(confidence):
    num_case = len(confidence)
    num_dataset = len(confidence[0][0])
    moc = [0 for _ in range(num_case)]
    for i in range(num_case):
        cases = list(range(num_case))
        other_cases = [x for x in cases if x != i]

        for k in range(num_dataset):
            current_conf = confidence[i][i][k]
            other_conf = []
            for other_case in other_cases:
                other_conf.append(confidence[other_case][i][k])
            max_other_conf = max(other_conf)
            moc[i] += current_conf - max_other_conf
        moc[i] = moc[i] / num_dataset
    return moc


warehouse = "500"

with open("converted_data_" + warehouse + "/causes.csv", "r") as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]


confidence = [[[] for _ in range(10)] for _ in range(10)]
fscore = [[[] for _ in range(10)] for _ in range(10)]


with open("single/confidence.txt", "rb") as fc:
    confidence = pickle.load(fc)

with open("single/fscore.txt", "rb") as ff:
    fscore = pickle.load(ff)

moc = calculate_moc(confidence)
# print(moc)
mfscore = calculate_mean_conf(fscore)
# print(mfscore)


x = np.arange(10)
plt.bar(x + 0.2, mfscore, width=0.4, label="f1-score")
plt.bar(x - 0.2, moc, width=0.4, label="margin")
plt.xticks(x, causes, rotation=70)
plt.yticks(np.arange(0, 90, step=10))
plt.title("Experiment 1: Accuracy of Single Causal Models")
plt.legend()
plt.show()

# %%
