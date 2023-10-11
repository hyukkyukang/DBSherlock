#%%

import heapq
from typing_extensions import final
import numpy as np
from matplotlib import pyplot as plt
import csv
import pickle
import statistics as stat
import dbsherlock_predicate_generation as p
import dbsherlock_single_causal_model as s
import dbsherlock_merged_causal_model as m
import pprint

num_case = 10
num_samples = 11
warehouse = '500'

with open("converted_data_"+warehouse+"/causes.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]
correct = []
incorrect = []
incorrect_case = [0 for _ in range(10)]
total_case = [0 for _ in range(10)]

for batch in range(num_samples):

    train_sample = [batch]
    test_sample = list(range(num_samples))
    for x in train_sample:
        test_sample.remove(x)
    
    for i in range(num_case):
        for j in test_sample:
            with open('single/explanation/{}_{}_{}.txt'.format(batch,i,j), 'rb') as fe:
                explanation = pickle.load(fe)

            explanation = [x for x in explanation if x != 0]
            explanation.sort(key=lambda x:-x[1])
            #print(explanation)
            total_case[i]+=1
            if explanation[0][0] == causes[i]:
                correct.append([i,batch,j])
                pprint.pprint("Correct case:{} train:{} test:{} ".format(i, batch, j))
            else:
                incorrect.append([i,batch,j])
                pprint.pprint("Incorrect case:{} train:{} test:{} ".format(i, batch, j))
                incorrect_case[i]+=1
print("correct: ")
pprint.pprint(correct)
print("incorrect: ")
pprint.pprint(incorrect)
pprint.pprint(incorrect_case)
pprint.pprint(total_case)
