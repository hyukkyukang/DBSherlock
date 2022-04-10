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

warehouse = str(500)
inf = math.inf

# for debugging
construct = 0
save= 0


with open("converted_data_"+warehouse+"/causes.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]


theta = 0.05
num_bins = 500
threshold_sp = 0.

num_train_sample = 5

num_case = 10
num_samples = 11
batch_count = 1


    
# Construct causal models
all_causal_models = [[] for i in range(num_case)]


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

def calculate_mean(data):
    result = []
    for stat in data:
        num_case = len(stat)
        mean_stat = [[0 for _ in range(num_case)] for _ in range(num_case)]

        for i in range(num_case):
            for j in range(num_case):
                temp = stat[i][j]
                mean_temp = stat.mean(temp)
        
        res_temp = [0 for i in range(num_case)]
        for i in range(num_case):
            res_temp[i] = mean_temp[i][i]
        
        result.append(res_temp)

    return result

def merge(all_causal_models, merge_idx):
    merged_causal_models = []
    for i in range(len(all_causal_models)):
        model = all_causal_models[i][merge_idx[0]]
        for idx in merge_idx[1:]:
            model_new = all_causal_models[i][idx]

            eps = {}
            preds = set(model.get_eps().keys()) & set(model_new.get_eps().keys())
            for pred in preds:
                pred_1 = model.get_eps()[pred]
                pred_2 = model_new.get_eps()[pred]
                if pred_1.type == 0:
                    type = 0
                    upper = max(pred_1.upper, pred_2.upper)
                    lower = min(pred_1.lower, pred_2.lower)
                    if upper < lower:
                        print("inconsistent predicate")
                        continue
                    predicate_string = ''

                    #assert ret_b >= ret_a, 'ret_b is less than ret_a'
                elif pred_1.type == 1: # Categorical
                    type = 1
                    c = set(pred_1.c,) | set(pred_2.c)
                attr_num, _, _, attr_name =pred_1.get_pred_info()
                eps[pred] = p.predicate(predicate_string, attr_num, attr_name, 0, type, lower, upper, c=0) #attr_num, attr_name, type, a, b, c=0
            model = p.causal_model(model.get_cv(), eps) 
        merged_causal_models.append(model)       
    return merged_causal_models


confidence = [[[] for _ in range(10)] for _ in range(10)]
fscore = [[[] for _ in range(10)] for _ in range(10)]
recall = [[[] for _ in range(10)] for _ in range(10)]
precision = [[[] for _ in range(10)] for _ in range(10)]
covered_normal_ratio = [[[] for _ in range(10)] for _ in range(10)]

# batch 하나에 대해서
if construct:
    for i in range(num_case):
        for j in range(num_samples):
            start_t = time.time()
            all_causal_models[i].append(p.causal_model(causes[i], p.predicate_generation(warehouse, i+1, j+1, num_bins, theta, threshold_sp)))
            end = time.time()
            t = end - start_t
            start_t = end
            #arr_time.append([' causal model generation : {} {}'.format(i,j),t])
            print([' causal model generation : {} {}'.format(i,j),t])
            continue
    if save:
        with open('merged/all_causal_models.txt', 'wb') as fa:
            pickle.dump(all_causal_models, fa)
else:
    with open('merged/all_causal_models.txt', 'rb') as fa:
        all_causal_models = pickle.load(fa) 

train_samples = []
with open('merged/train_sample_list_{}.txt'.format(num_train_sample)) as fs:
    data = fs.readlines()
for x in data:
    x_list = x.rstrip('\n').split()
    x_list = list(map(lambda y:int(y)-1, x_list))
    train_samples.append(x_list)
    



for batch in range(batch_count):
    train_sample = train_samples[batch]
    
    test_sample = list(range(num_samples))
    for i in train_sample:
        test_sample.remove(i)

    
    ###################################################################
    # merge the causal models in train sample list of this batch!!!   #
    ###################################################################
    merged_causal_models=merge(all_causal_models, train_sample) # 각 배치마다 merged causal model을 만들기 1*10
    if batch == 0:
        with open('merged/merged_causal_models_{}.txt'.format(num_train_sample), 'wb') as fa:
            pickle.dump(merged_causal_models, fa)            
    
    for i in range(num_case):
        for j in test_sample:
            
            # explanation : 10*5

            print("batch : {} i : {} j : {}".format(batch,i,j))
            explanation = [[] for i in range(num_case)]
            num_attr, attr_name, n, ab, d, n_index, ab_index, timestamp = p.load_data(warehouse, i+1, j+1)
            for k, c in enumerate(merged_causal_models):
                explanation[k] = c.cal_confidence(n, ab, d, num_bins, i+1, j+1) 
            if save:
                with open('merged/explanation/{}_{}_{}_{}.txt'.format(num_train_sample, batch,i,j), 'wb') as fe:
                    pickle.dump(explanation, fe)


            #print(['test : {} {} cal confidence'.format(i,j),t])   

            for id, ex in enumerate(explanation):
                if ex == 0:
                    print(i, k, id,"ex is zero")
            explanation = [x for x in explanation if x != 0]
            explanation.sort(key=lambda x:-x[1])
            #print(explanation)


            for k in range(num_case):
                idxes = [x for x in range(len(explanation)) if explanation[x][0] == causes[k]]
                if len(idxes) is not 0:
                    idx = idxes[0]
                    recall[k][i].append(explanation[idx][3])
                    precision[k][i].append(explanation[idx][2])
                    #covered_normal_ratio[k][i].append(explanation[idx][5])

                    confidence[k][i].append(explanation[idx][1])
                    fscore[k][i].append(explanation[idx][4])
                else:
                    print(i, k,"pass")


            #print("first",explanation[0][0], explanation[0][1])
            #print("second",explanation[1][0], explanation[1][1])

    #print(['test',t])
    #print(confidence)
    #print(fscore)
if save:
    with open('merged/explanation_{}.txt'.format(num_train_sample), 'wb') as f:
        pickle.dump(explanation, f)

    with open('merged/confidence_{}.txt'.format(num_train_sample), 'wb') as f:
        pickle.dump(confidence, f)

    with open('merged/fscore_{}.txt'.format(num_train_sample), 'wb') as f:
        pickle.dump(fscore, f)

    with open('merged/recall_{}.txt'.format(num_train_sample), 'wb') as f:
        pickle.dump(recall, f)

    with open('merged/precision_{}.txt'.format(num_train_sample), 'wb') as f:
        pickle.dump(precision, f)

    with open('merged/covered_normal_ratio_{}.txt'.format(num_train_sample), 'wb') as f:
        pickle.dump(covered_normal_ratio, f)


with open('single/confidence.txt', 'rb') as fc:
    confidence_s = pickle.load(fc)

with open('single/fscore.txt', 'rb') as ff:
    fscore_s = pickle.load(ff)



with open('back_up/merged_10/confidence.txt', 'rb') as fc:
    confidence_10 = pickle.load(fc)

with open('back_up/merged_10/fscore.txt', 'rb') as ff:
    fscore_10 = pickle.load(ff)

moc_10 = calculate_moc(confidence_10)
moc_s = calculate_moc(confidence_s)
moc = calculate_moc(confidence)
#print(moc)
mfscore_10 = calculate_mean_conf(fscore_10)
mfscore_s = calculate_mean_conf(fscore_s)
mfscore = calculate_mean_conf(fscore)
#print(mfscore)





plt.figure()
x = np.arange(10)
plt.bar(x-0.2, moc_s, width = 0.4, label = 'single')
plt.bar(x+0.2, moc, width = 0.4, label = 'merged-5')
plt.xticks(x, causes, rotation = 45)
plt.yticks(np.arange(0, 70, step=10))
plt.title('Effectiveness of Merged Causal Models : Margin of Confidence')

plt.legend()



plt.figure()
x = np.arange(10)
plt.bar(x-0.2, moc_s, width = 0.2, label = 'single')
plt.bar(x, moc, width = 0.2, label = 'merged-5')
plt.bar(x+0.2, moc_10, width = 0.2, label = 'merged-10')
plt.xticks(x, causes, rotation = 45)
plt.yticks(np.arange(0, 70, step=10))
plt.title('Effectiveness of Merged Causal Models : Margin of Confidence')

plt.legend()


plt.figure()
plt.bar(x-0.2, mfscore_s, width = 0.2, label = 'single')
plt.bar(x, mfscore, width = 0.2, label = 'merged-5')
plt.bar(x+0.2, mfscore_10, width = 0.2, label = 'merged-10')
plt.xticks(x, causes, rotation = 45)
plt.yticks(np.arange(0, 70, step=10))
plt.title('Effectiveness of Merged Causal Models : Mean F-score')
plt.legend()

plt.show()



  # %%
 