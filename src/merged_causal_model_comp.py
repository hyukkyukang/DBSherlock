# %%

import heapq
from typing_extensions import final
import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pickle
import statistics as stat
import dbsherlock_predicate_generation as p

import time

warehouse = str(500)
inf = math.inf

# for debugging
construct = 0
save = 0


with open("converted_data_" + warehouse + "/causes.csv", "r") as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]


theta = 0.05
num_bins = 500
threshold_sp = 0.0


num_case = 10
num_samples = 11
batch_count = 1


with open("single/all_causal_models.txt", "rb") as f:
    all_causal_models = pickle.load(f)

with open("merged/merged_causal_models_5.txt", "rb") as f:
    merged_causal_models_5 = pickle.load(f)

with open("merged/merged_causal_models_10.txt", "rb") as f:
    merged_causal_models_10 = pickle.load(f)


single_causal_models = [
    x[0] for x in all_causal_models
]  # 0번째 dataset으로 만들어진 causal model

for i in range(10):
    print("Single")
    for x in single_causal_models[i].get_eps().values():
        n, lower, upper, attr_name = x.get_pred_info()
        print("[{}] {:.2f} < {} < {:.2f}".format(n, lower, attr_name, upper))

    print("Merged_5")
    print(merged_causal_models_5[i].get_cv())
    for x in merged_causal_models_5[i].get_eps().values():
        n, lower, upper, attr_name = x.get_pred_info()
        print("[{}] {:.2f} < {} < {:.2f}".format(n, lower, attr_name, upper))

    print("Merged_10")
    print(merged_causal_models_10[i].get_cv())
    for x in merged_causal_models_10[i].get_eps().values():
        n, lower, upper, attr_name = x.get_pred_info()
        print("[{}] {:.2f} < {} < {:.2f}".format(n, lower, attr_name, upper))

# %%
with open("single/explanation.txt", "rb") as f:
    single_explanation = pickle.load(f)

with open("merged/explanation_5.txt", "rb") as f:
    merge_5_explanation = pickle.load(f)

with open("merged/explanation_10.txt", "rb") as f:
    merge_10_explanation = pickle.load(f)
