import dbsherlock_predicate_generation as p
import heapq
import numpy as np
from matplotlib import pyplot as plt



 # Prediction
def merged_predict(warehouse, causal_model, merge_idx, model_i, causes): # merge_idx : merge 할 때 사용한 
    conf_index = list(range(11))
    for i in merge_idx:
        conf_index.remove(i)

    result = 0
    # (i, j) = causal_model_idx
    confidence = [[] for _ in range(10)]
    for i in range(10):
        for j in conf_index:
            num_attr, attr_name, n, ab, d, n_index, ab_index, timestamp = p.load_data(i+1, j+1)
            confidence[i].append(causal_model.cal_confidence(n, ab))
    confidence = np.array(confidence)
    avg_conf = [0]*10
    for i in range(10):
        avg_conf[i] = confidence[i].sum()/len(conf_index)

    index_causes = heapq.nlargest(2, range(len(avg_conf)), key=avg_conf.__getitem__)

    #print("<<The probable root causes of case {}>>".format(model_i))
    margin = avg_conf[index_causes[0]]-avg_conf[index_causes[1]]

    #print("The 1st probable root cause : {}".format(causes[index_causes[0]]))
    #print("The 2nd probable root cause : {}".format(causes[index_causes[1]]))
    #print("The margin : {}".format(margin))
    if index_causes[0] == model_i:
        result = 1
       # print("Correct!")
    #else:
       # print("Incorrect!")
   # print("")
    return result, margin, index_causes[0]

 
def merge(model_1, model_2):
    ret_type = -1
    ret_a = -1
    ret_b = -1
    ret_c = -1
    ret_eps = {}
    preds = set(model_1.get_eps().keys()) & set(model_2.get_eps().keys())
    for pred in preds:
        pred_1 = model_1.get_eps()[pred]
        pred_2 = model_2.get_eps()[pred]
        if pred_1.type == pred_2.type:
            if pred_1.type == 0:
                ret_type = 0
                ret_b = max(pred_1.b, pred_2.b)
            elif pred_1.type == 1:
                ret_type = 1
                ret_a = min(pred_1.a, pred_2.a)
            elif pred_1.type == 2:
                ret_type = 2
                ret_b = max(pred_1.b, pred_2.b)
                ret_a = min(pred_1.a, pred_2.a)
                if ret_b >= ret_a:
                    print("inconsistent predicate")
                    continue
                #assert ret_b >= ret_a, 'ret_b is less than ret_a'
            elif pred_1.type == 3: # Categorical
                ret_type = 3
                ret_c = set(pred_1.c,) | set(pred_2.c)
        else : 
            ret_type = 2
            ret_b = max(pred_1.b, pred_2.b)
            ret_a = min(pred_1.a, pred_2.a)
            if ret_b >= ret_a:
                print("inconsistent predicate")
                continue
            # assert ret_b >= ret_a, 'ret_b is less than ret_a'
        ret_eps[pred] = p.predicate(len(preds), pred, ret_type, ret_a, ret_b, ret_c) #attr_num, attr_name, type, a, b, c=0
    ret = p.causal_model(model_1.get_cv(), ret_eps) 
    #print(ret_eps)   
    return ret



