import csv
import pickle
import pprint

import numpy as np
from matplotlib import pyplot as plt

import dbsherlock_predicate_generation as p

# %%

num_attr, attr_name, n_6, ab_6, d_6, n_index_6, ab_index_6, timestamp = p.load_data(
    "500", 6, 1
)
_, _, n_8, ab_8, d_8, n_index_8, ab_index_8, _ = p.load_data("500", 8, 1)


x = np.array(timestamp)

for i in range(20):
    plt.figure()
    plt.plot(n_index_6, n_6[i], "gv", label="normal_6")
    plt.plot(ab_index_6, ab_6[i], "rv", label="abnormal_6")

    if np.array_equal(n_6[i], n_8[i]):
        print("normal{} same".format(i))

    if np.array_equal(ab_6[i], ab_8[i]):
        print("abnormal{} same".format(i))
    plt.plot(n_index_8, n_8[i], "g^", label="normal_8")
    plt.plot(ab_index_8, ab_8[i], "r^", label="abnormal_8")
    plt.legend()
    plt.title("{}th attribute: {}".format(i, attr_name[i]))

plt.show()


# %%

num_case = 10
num_samples = 11
warehouse = "500"

with open("converted_data_" + warehouse + "/causes.csv", "r") as f:
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
            with open(
                "single/explanation/{}_{}_{}.txt".format(batch, i, j), "rb"
            ) as fe:
                explanation = pickle.load(fe)

            explanation = [x for x in explanation if x != 0]
            explanation.sort(key=lambda x: -x[1])
            # print(explanation)
            total_case[i] += 1
            if explanation[0][0] == causes[i]:
                correct.append([i, batch, j])
                pprint.pprint("Correct case:{} train:{} test:{} ".format(i, batch, j))
            else:
                incorrect.append([i, batch, j])
                pprint.pprint("Incorrect case:{} train:{} test:{} ".format(i, batch, j))
                incorrect_case[i] += 1
print("correct: ")
pprint.pprint(correct)
print("incorrect: ")
pprint.pprint(incorrect)
pprint.pprint(incorrect_case)
pprint.pprint(total_case)

# %%

a = 4
b = 8

print("a: {} b: {}".format(a, b))

for j in range(11):
    print()
    print(j, "th dataset")
    with open("single/all_causal_models.txt", "rb") as fa:
        all_causal_models = pickle.load(fa)
    attr_a = []  # 72
    for x in all_causal_models[a][j].get_eps().values():
        n, _, _, _ = x.get_pred_info()
        attr_a.append(n)

    attr_b = []  # 105
    for x in all_causal_models[b][j].get_eps().values():
        n, _, _, _ = x.get_pred_info()
        attr_b.append(n)

    diff_a = []
    diff_b = []
    only_a = []
    only_b = []
    intersect = list(set(attr_a) & set(attr_b))
    intersect.sort()

    print("겹치는 attribute")
    print(intersect, len(intersect))

    print("case a에만 존재하는 특징적인 predicate")
    for x in all_causal_models[a][j].get_eps().values():
        n, _, _, _ = x.get_pred_info()
        if n in intersect:
            diff_a.append(np.round(x.get_diff(), 2))
        else:
            only_a.append(np.round(x.get_diff(), 2))
            if x.get_diff() > 0.5:
                x.print_pred()
        attr_a.append(n)

    attr_b = []  # 105
    print("case b에만 존재하는 특징적인 predicate")

    for x in all_causal_models[b][j].get_eps().values():
        n, _, _, _ = x.get_pred_info()
        if n in intersect:
            diff_b.append(np.round(x.get_diff(), 2))
        else:
            only_b.append(np.round(x.get_diff(), 2))
            if x.get_diff() > 0.5:
                x.print_pred()
        attr_b.append(n)

    print("겹치는 attribute들의 predicate의 diff값")
    print(diff_a, diff_b)

    print("안 겹치는 attribute들의 predicate의 diff값")
    print(only_a, only_b)
# intersect되는 attribute들이 diff가 얼마나 높은지
# predicate의 영역은 실제로 얼마나 겹치는지
# %%
with open("single/all_causal_models.txt", "rb") as fa:
    all_causal_models = pickle.load(fa)

all_causal_models[0][0].print_causal_model()
# %%
# Analysis of all causal models
j = 0  # dataset
with open("single/all_causal_models.txt", "rb") as fa:
    all_causal_models = pickle.load(fa)
num_case = len(all_causal_models)
all_diff_preds = [[[] for _ in range(num_case)] for _ in range(num_case)]
count = [{} for i in range(num_case)]

for a in range(num_case):
    for b in range(num_case):
        if a >= b:
            continue
        attr_a = []
        for x in all_causal_models[a][j].get_eps().values():
            n, _, _, _ = x.get_pred_info()
            attr_a.append(n)

        attr_b = []
        for x in all_causal_models[b][j].get_eps().values():
            n, _, _, _ = x.get_pred_info()
            attr_b.append(n)

        intersect = list(set(attr_a) & set(attr_b))
        intersect.sort()

        # print("겹치는 attribute")
        # print(intersect, len(intersect))

        print("case{}에만 존재하고 {}에 없는 diff 0.5 이상 predicate".format(a, b))
        for x in all_causal_models[a][j].get_eps().values():
            n, _, _, _ = x.get_pred_info()
            if n not in intersect and x.get_diff() > 0.5:
                all_diff_preds[a][b].append([n, np.round(x.get_diff(), 2)])
                x.print_pred()
                if n in count[a]:
                    count[a][n] += 1
                else:
                    count[a][n] = 1

        print("case{}에만 존재하고 {}에 없는 diff 0.5 이상 predicate".format(b, a))
        for x in all_causal_models[b][j].get_eps().values():
            n, _, _, _ = x.get_pred_info()
            if n not in intersect and x.get_diff() > 0.5:
                all_diff_preds[b][a].append([n, np.round(x.get_diff(), 2)])
                x.print_pred()
                if n in count[b]:
                    count[b][n] += 1
                else:
                    count[b][n] = 1

# pprint.pprint(all_diff_preds)
print(count)

over_7_count = [
    dict(filter(lambda elem: elem[1] >= 7, count[i].items())) for i in range(num_case)
]


print(over_7_count)

# %%
from collections import Counter

with open("single/all_causal_models.txt", "rb") as fa:
    all_causal_models = pickle.load(fa)

for i in range(10):
    print("i : {}".format(i))
    attr = []
    for j in range(11):
        for x in all_causal_models[i][j].get_eps().values():
            n, _, _, _ = x.get_pred_info()
            attr.append(n)

    cnt = Counter(attr)
    print("가장 많이 있는 원소 차례대로 10개 뽑기 :\n", cnt.most_common(10))

    for count in cnt.most_common(10):
        for j in range(11):
            for x in all_causal_models[i][j].get_eps().values():
                n, _, _, _ = x.get_pred_info()
                if count[0] == n:
                    x.print_pred()
                else:
                    continue
    print()

# %%

# simple intersection
with open("single/all_causal_models.txt", "rb") as fa:
    all_causal_models = pickle.load(fa)

for i in range(10):
    print("i : {}".format(i))

    for j in range(11):
        attr = []
        for x in all_causal_models[i][j].get_eps().values():
            n, _, _, _ = x.get_pred_info()
            attr.append(n)
        if j == 0:
            intersect = attr
        else:
            intersect = list(set(attr) & set(intersect))
            intersect.sort()

    print("겹치는 attribute")
    print(intersect, len(intersect))

    print()

# %%
