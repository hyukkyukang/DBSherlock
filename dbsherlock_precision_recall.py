#%%

import heapq
from typing_extensions import final
import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pickle
import statistics as stat

import time

warehouse = str(500)



with open("converted_data_"+warehouse+"/causes.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]


def calculate_mean_pr(precision, recall):
    
    num_case = len(precision)
    mean_pre = [[0 for _ in range(num_case)] for _ in range(num_case)]
    mean_rec = [[0 for _ in range(num_case)] for _ in range(num_case)]

    for i in range(num_case):
        for j in range(num_case):
            pre = precision[i][j]
            mean_pre[i][j] = stat.mean(pre)
            rec = recall[i][j]
            mean_rec[i][j] = stat.mean(rec)

    
    final_pre =  [0 for _ in range(num_case)]
    final_rec =  [0 for _ in range(num_case)]
    for i in range(num_case):
        final_pre[i] = mean_pre[i][i]
        final_rec[i] = mean_rec[i][i]

    return final_pre, final_rec


def calculate_mean(data):
    result = []
    for d in data:
        num_case = len(d)
        mean_d = [[0 for _ in range(num_case)] for _ in range(num_case)]

        for i in range(num_case):
            for j in range(num_case):
                temp = d[i][j]
                mean_d[i][j] = stat.mean(temp)
        
        res_d = [0 for i in range(num_case)]
        for i in range(num_case):
            res_d[i] = mean_d[i][i]
        
        result.append(res_d)

    return result

with open('single/precision.txt', 'rb') as f:
    pre_s = pickle.load(f)

with open('single/recall.txt', 'rb') as f:
    rec_s = pickle.load(f)

with open('single/covered_normal_ratio.txt', 'rb') as f:
    covered_normal_ratio_s = pickle.load(f)

with open('merged/precision_5.txt', 'rb') as f:
    pre_5 = pickle.load(f)

with open('merged/recall_5.txt', 'rb') as f:
    rec_5 = pickle.load(f)

with open('merged/covered_normal_ratio_5.txt', 'rb') as f:
    covered_normal_ratio_5 = pickle.load(f)

with open('merged/precision_10.txt', 'rb') as f:
    pre_10 = pickle.load(f)

with open('merged/recall_10.txt', 'rb') as f:
    rec_10 = pickle.load(f)

with open('merged/covered_normal_ratio_10.txt', 'rb') as f:
    covered_normal_ratio_10 = pickle.load(f)

pre_s, rec_s = calculate_mean_pr(pre_s, rec_s)
pre_5, rec_5 = calculate_mean_pr(pre_5, rec_5)
pre_10, rec_10 = calculate_mean_pr(pre_10, rec_10)

covered_normal_ratio_average = calculate_mean([covered_normal_ratio_s,covered_normal_ratio_5, covered_normal_ratio_10])

plt.figure()
x = np.arange(10)
plt.bar(x-0.2, covered_normal_ratio_average[0], width = 0.2, label = 'single')
plt.bar(x, covered_normal_ratio_average[1], width = 0.2, label = 'merged-5')
plt.bar(x+0.2, covered_normal_ratio_average[2], width = 0.2, label = 'merged-10')
plt.xticks(x, causes, rotation = 70)
plt.yticks(np.arange(0, 110, step=10))
plt.title('Covered Normal Ratio of Single/Merged Causal Models')
plt.legend()


plt.figure()
plt.bar(x-0.2, rec_s, width = 0.2, label = 'single')
plt.bar(x, rec_5, width = 0.2, label = 'merged-5')
plt.bar(x+0.2, rec_10, width = 0.2, label = 'merged-10')
plt.xticks(x, causes, rotation = 70)
plt.yticks(np.arange(0, 110, step=10))
plt.title('Recall of Single/Merged Causal Models')
plt.legend()

plt.show()



# %%
