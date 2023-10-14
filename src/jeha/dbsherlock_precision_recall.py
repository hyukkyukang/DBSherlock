import csv
import pickle

import numpy as np
from matplotlib import pyplot as plt

from src.utils import calculate_mean, calculate_mean_pr

warehouse = str(500)

file_path = "data/converted_data_" + warehouse + "/causes.csv"
with open(file_path, "r") as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]


with open("single/precision.txt", "rb") as f:
    pre_s = pickle.load(f)

with open("single/recall.txt", "rb") as f:
    rec_s = pickle.load(f)

with open("single/covered_normal_ratio.txt", "rb") as f:
    covered_normal_ratio_s = pickle.load(f)

with open("merged/precision_5.txt", "rb") as f:
    pre_5 = pickle.load(f)

with open("merged/recall_5.txt", "rb") as f:
    rec_5 = pickle.load(f)

with open("merged/covered_normal_ratio_5.txt", "rb") as f:
    covered_normal_ratio_5 = pickle.load(f)

with open("merged/precision_10.txt", "rb") as f:
    pre_10 = pickle.load(f)

with open("merged/recall_10.txt", "rb") as f:
    rec_10 = pickle.load(f)

with open("merged/covered_normal_ratio_10.txt", "rb") as f:
    covered_normal_ratio_10 = pickle.load(f)

pre_s, rec_s = calculate_mean_pr(pre_s, rec_s)
pre_5, rec_5 = calculate_mean_pr(pre_5, rec_5)
pre_10, rec_10 = calculate_mean_pr(pre_10, rec_10)

covered_normal_ratio_average = calculate_mean(
    [covered_normal_ratio_s, covered_normal_ratio_5, covered_normal_ratio_10]
)

plt.figure()
x = np.arange(10)
plt.bar(x - 0.2, covered_normal_ratio_average[0], width=0.2, label="single")
plt.bar(x, covered_normal_ratio_average[1], width=0.2, label="merged-5")
plt.bar(x + 0.2, covered_normal_ratio_average[2], width=0.2, label="merged-10")
plt.xticks(x, causes, rotation=70)
plt.yticks(np.arange(0, 110, step=10))
plt.title("Covered Normal Ratio of Single/Merged Causal Models")
plt.legend()


plt.figure()
plt.bar(x - 0.2, rec_s, width=0.2, label="single")
plt.bar(x, rec_5, width=0.2, label="merged-5")
plt.bar(x + 0.2, rec_10, width=0.2, label="merged-10")
plt.xticks(x, causes, rotation=70)
plt.yticks(np.arange(0, 110, step=10))
plt.title("Recall of Single/Merged Causal Models")
plt.legend()

plt.show()
