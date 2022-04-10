#%%

import heapq
from typing_extensions import final
import numpy as np
from matplotlib import pyplot as plt
import csv
import pickle
import statistics as stat
import dbsherlock_predicate_generation as p
import itertools

warehouse = str(500)

# for debugging
construct = 0
save = 0


with open("converted_data_"+warehouse+"/causes.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))
causes = data[0]


theta = 0.2
num_bins = 500
threshold_sp = 0.0


num_case = 10
num_samples = 11

    
# Construct causal models
all_causal_models = [[] for i in range(num_case)]


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
        if num_dataset==0:
            moc[i] = 0
        else:
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


recall = [[[] for _ in range(10)] for _ in range(10)]
precision = [[[] for _ in range(10)] for _ in range(10)]
confidence = [[[] for _ in range(10)] for _ in range(10)]
fscore = [[[] for _ in range(10)] for _ in range(10)]
covered_normal_ratio = [[[] for _ in range(10)] for _ in range(10)]

# batch 하나에 대해서
if construct:
    for i in range(num_case):
        for j in range(num_samples):
            all_causal_models[i].append(p.causal_model(causes[i], p.predicate_generation(warehouse, i+1, j+1, num_bins, theta, threshold_sp)))

    #with open('all_causal_models.txt', 'wb') as fa:
        #pickle.dump(all_causal_models, fa)
else:
    with open('single/all_causal_models.txt', 'rb') as fa:
        all_causal_models = pickle.load(fa) 

filtered_count = [[] for _ in range(num_samples)]

for j in range(num_samples):

    all_diff_preds = [[[] for _ in range(num_case)] for _ in range(num_case)]
    count = [{} for i in range(num_case)]

    for a in range(num_case):
        for b in range(num_case):
            if a >= b:
                continue
            attr_a = [] 
            for x in all_causal_models[a][j].get_eps().values():
                n,_,_,_ = x.get_pred_info()
                attr_a.append(n)

            attr_b = [] 
            for x in all_causal_models[b][j].get_eps().values():
                n,_,_,_ = x.get_pred_info()
                attr_b.append(n)


            intersect = list(set(attr_a) & set(attr_b))
            intersect.sort()

            #print("겹치는 attribute")
            #print(intersect, len(intersect))

            #print("case{}에만 존재하고 {}에 없는 diff 0.5 이상 predicate".format(a,b))
            for x in all_causal_models[a][j].get_eps().values():
                n,_,_,_ = x.get_pred_info()
                if n not in intersect and x.get_diff()>0.5:
                    all_diff_preds[a][b].append([n, np.round(x.get_diff(),2)])
                    #x.print_pred()
                    if n in count[a]:
                        count[a][n] += 1
                    else:
                        count[a][n] = 1

            #print("case{}에만 존재하고 {}에 없는 diff 0.5 이상 predicate".format(b,a))
            for x in all_causal_models[b][j].get_eps().values():
                n,_,_,_ = x.get_pred_info()
                if n not in intersect and x.get_diff()>0.5:
                    all_diff_preds[b][a].append([n, np.round(x.get_diff(),2)])
                    #x.print_pred()
                    if n in count[b]:
                        count[b][n]+= np.round(x.get_diff(),2)
                    else:
                        count[b][n]= np.round(x.get_diff(),2) 

    #pprint.pprint(all_diff_preds)
    #print(count)
    for i in range(num_case):
        count[i] = dict(sorted(count[i].items(), key = lambda item: item[1], reverse = True)[:5])

        #count[i] = dict(itertools.islice(count[i].items(), 5))
    filtered_count[j] = [dict(filter(lambda elem:elem[1]>=7, count[i].items())) for i in range(num_case)]
    # j번째 dataset에서 얻은 각 test case의 특징적인 attribute
    # j번째 batch에 input으로 들어감
  
    #print(filtered_count)






for batch in range(5,6):
    
    train_sample = [batch]
    test_sample = list(range(num_samples))
    for i in train_sample:
        test_sample.remove(i)
            
    
    for i in range(num_case):
        #if i != 5 and i != 7:
            #continue
        for j in test_sample:
            
            print("batch : {} i: {} j :{}".format(batch, i, j))
            # explanation : 10*5

            explanation = [[] for i in range(num_case)]
            num_attr, attr_name, n, ab, d, n_index, ab_index, timestamp = p.load_data(warehouse, i+1, j+1)

            for k, c in enumerate(all_causal_models):
                c = c[train_sample[0]]
                explanation[k] = c.cal_confidence_test(n, ab, d, num_bins, i+1, j+1, filtered_count[batch]) 
            
            if 0:
                with open('single/explanation/{}_{}_{}.txt'.format(batch,i,j), 'wb') as fe:
                    pickle.dump(explanation, fe)


            #print(['test : {} {} cal confidence'.format(i,j),t])   

            for id, ex in enumerate(explanation):
                if ex == 0:
                    print(i, k, id,"ex is zero")
            explanation = [x for x in explanation if x != 0]
            explanation.sort(key=lambda x:-x[1])
            #print(explanation)

            

            #print(['test : {} {} after sorting'.format(i,j),t])

            for k in range(num_case):
                idxes = [x for x in range(len(explanation)) if explanation[x][0] == causes[k]]
                if len(idxes) is not 0:
                    idx = idxes[0]
                    recall[k][i].append(explanation[idx][3])
                    precision[k][i].append(explanation[idx][2])
                    confidence[k][i].append(explanation[idx][1])
                    fscore[k][i].append(explanation[idx][4])
                    covered_normal_ratio[k][i].append(explanation[idx][5])
                else:
                    print(i, k,"pass")


            
            print("first",explanation[0][0], explanation[0][1])
            print("second",explanation[1][0], explanation[1][1])

    #print(['test',t])
    #print(confidence)
    #print(fscore)

if save:
    with open('single/confidence.txt', 'wb') as f:
        pickle.dump(confidence, f)

    with open('single/fscore.txt', 'wb') as f:
        pickle.dump(fscore, f)

    with open('single/recall.txt', 'wb') as f:
        pickle.dump(recall, f)

    with open('single/precision.txt', 'wb') as f:
        pickle.dump(precision, f)
    
    with open('single/covered_normal_ratio.txt', 'wb') as f:
        pickle.dump(covered_normal_ratio, f)
        

moc = calculate_moc(confidence)
print(moc)
mfscore = calculate_mean_conf(fscore)
print(mfscore)


x = np.arange(10)
plt.bar(x+0.2, mfscore, width = 0.4, label = 'f1-score')
plt.bar(x-0.2, moc, width = 0.4, label = 'margin')
plt.xticks(x, causes, rotation = 70)
plt.yticks(np.arange(0, 90, step=10))
plt.legend()
plt.title('Experiment 1: Accuracy of Single Causal Models')

plt.show() 




 # %%
 