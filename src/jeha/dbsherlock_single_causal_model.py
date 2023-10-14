import heapq

import numpy as np

import src.dbsherlock_predicate_generation as p


# Prediction
def single_predict(warehouse, causal_model, model_i, model_j, causes, num_bins):
    result = 0
    # (i, j) = causal_model_idx
    test_sample = list(range(11)).remove(model_j)
    confidence = [[] for _ in range(10)]
    for i in range(10):  # num_Case
        for j in test_sample:  # test sample
            num_attr, attr_name, n, ab, d, n_index, ab_index, timestamp = p.load_data(
                warehouse, i + 1, j + 1
            )
            causal_model
            confidence[i].append(causal_model.cal_confidence(n, ab, num_bins))

    confidence = np.array(confidence)
    # print(confidence)
    avg_conf = [0] * 10
    for i in range(10):
        if model_i == i:
            avg_conf[i] = confidence[i].sum() / 10
        else:
            avg_conf[i] = confidence[i].sum() / 11

    index_causes = heapq.nlargest(2, range(len(avg_conf)), key=avg_conf.__getitem__)

    print("<<The probable root causes of case {}>>".format(model_i))
    margin = avg_conf[index_causes[0]] - avg_conf[index_causes[1]]

    print("The 1st probable root cause : {}".format(causes[index_causes[0]]))
    print("The 2nd probable root cause : {}".format(causes[index_causes[1]]))
    print("The margin : {}".format(margin))
    if index_causes[0] == model_i:
        result = 1
        # print("Correct!")
    #    else:
    # print("Incorrect!")
    # print("")

    return result, margin, index_causes[0]
