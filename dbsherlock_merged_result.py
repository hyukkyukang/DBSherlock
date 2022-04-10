#%%


import heapq
from typing_extensions import final
import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pickle
import statistics as stat
import dbsherlock_predicate_generation as p
import dbsherlock_single_causal_model as s
import dbsherlock_merged_causal_model as m
import time

warehouse='500'

with open("converted_data_"+warehouse+"/causes.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]



def calculate_moc(confidence):
    num_case = len(confidence)
    num_dataset = len(confidence[0][0])
    moc = [0 for _ in range(num_case)]
    for i in range(num_case):
        cases = list(range(num_case))
        other_cases =[x for x in cases if x != i]
        

        for k in range(num_dataset):
            current_conf = confidence[i][i][k]
            other_conf = []
            for other_case in other_cases:
                other_conf.append(confidence[other_case][i][k])
            max_other_conf = max(other_conf)
            moc[i] += (current_conf - max_other_conf)
        moc[i] = moc[i] / num_dataset
    return moc

def calculate_mean_conf(confidence):
    
    num_case = len(confidence)
    mean_conf = [[0 for _ in range(num_case)] for _ in range(num_case)]
    for i in range(num_case):
        for j in range(num_case):
            conf = confidence[i][j]
            mean_conf[i][j] = stat.mean(conf)
    
    final_conf =  [0 for _ in range(num_case)]
    for i in range(num_case):
        final_conf[i] = mean_conf[i][i]

    return final_conf


with open('single/confidence.txt', 'rb') as fc:
    confidence_s = pickle.load(fc)

with open('single/fscore.txt', 'rb') as ff:
    fscore_s = pickle.load(ff)



with open('merged/confidence.txt', 'rb') as fc:
    confidence = pickle.load(fc)

with open('merged/fscore.txt', 'rb') as ff:
    fscore = pickle.load(ff)


moc_s = calculate_moc(confidence_s)
moc = calculate_moc(confidence)
#print(moc)
mfscore = calculate_mean_conf(fscore)
#print(mfscore)




x = np.arange(10)
plt.bar(x-0.2, moc_s, width = 0.4, label = 'single')
plt.bar(x+0.2, moc, width = 0.4, label = 'merged')
plt.xticks(x, causes, rotation = 70)
plt.yticks(np.arange(0, 70, step=10))
plt.title('Effectiveness of Merged Causal Models')
plt.legend()
plt.show() 
# %%
