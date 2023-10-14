import csv
import pickle

import numpy as np
from matplotlib import pyplot as plt

import dbsherlock_merged_causal_model as m
import dbsherlock_predicate_generation as p
import dbsherlock_single_causal_model as s
from src.utils import calculate_mean_conf


def calculate_moc(confidence):
    num_case = len(confidence)
    num_dataset = len(confidence[0][0])
    moc = [0 for _ in range(num_case)]
    for i in range(num_case):
        cases = list(range(num_case))
        other_cases = [x for x in cases if x != i]

        for k in range(num_dataset):
            current_conf = confidence[i][i][k]
            other_conf = []
            for other_case in other_cases:
                other_conf.append(confidence[other_case][i][k])
            max_other_conf = max(other_conf)
            moc[i] += current_conf - max_other_conf
        moc[i] = moc[i] / num_dataset
    return moc


def main():
    number = "2"
    warehouse = str(500)

    # for debugging
    construct = 0

    with open("data/converted_data_" + warehouse + "/causes.csv", "r") as f:
        data = list(csv.reader(f, delimiter=","))
    causes = data[0]

    theta = 0.05
    num_bins = 500
    threshold_sp = 0.0

    num_case = 10
    num_samples = 50

    # Construct causal models

    all_causal_models = [[] for i in range(num_case)]
    # if construct == 1:
    #     with open('etc/merged/all_causal_models.txt', 'wb') as fa:
    #         pickle.dump(all_causal_models, fa)
    #     exit()
    # else:
    #     with open('etc/merged/all_causal_models.txt', 'rb') as fa:
    #         all_causal_models = pickle.load(fa)

    confidence = [[[] for _ in range(10)] for _ in range(10)]
    fscore = [[[] for _ in range(10)] for _ in range(10)]

    for batch in range(num_samples):
        train_sample = [batch]
        test_sample = list(range(num_samples))
        for i in train_sample:
            test_sample.remove(i)

        for i in range(num_case):
            for j in train_sample:
                all_causal_models[i].append(
                    p.causal_model(
                        causes[i],
                        p.predicate_generation(
                            warehouse, i + 1, j + 1, num_bins, theta, threshold_sp
                        ),
                    )
                )

        for i in range(num_case):
            for j in test_sample:
                # explanation : 10*5
                print("batch : {} i : {} j : {}".format(batch, i, j))
                explanation = [[] for i in range(num_case)]
                (
                    num_attr,
                    attr_name,
                    n,
                    ab,
                    d,
                    n_index,
                    ab_index,
                    timestamp,
                ) = p.load_data(warehouse, i + 1, j + 1)
                causal_models = [x[j] for x in all_causal_models]

                for k, c in enumerate(all_causal_models):
                    c = c[train_sample[0]]

                    # print("{} {}".format(i+1,j+1))
                    explanation[k] = c.cal_confidence(n, ab, d, num_bins, i + 1, j + 1)

                for id, ex in enumerate(explanation):
                    if ex == 0:
                        print(i, k, id, "ex is zero")
                explanation = [x for x in explanation if x != 0]
                explanation.sort(key=lambda x: -x[1])
                # print(explanation)
                for k in range(num_case):
                    idxes = [
                        x
                        for x in range(len(explanation))
                        if explanation[x][0] == causes[k]
                    ]
                    if len(idxes) != 0:
                        idx = idxes[0]
                        confidence[k][i].append(explanation[idx][1])
                        fscore[k][i].append(explanation[idx][4])
                    else:
                        print(i, k, "pass")

                # print("first",explanation[0][0], explanation[0][1])
                # print("second",explanation[1][0], explanation[1][1])
        # print(confidence)
        # print(fscore)
    if number == "1":
        with open("explanation.txt", "wb") as fe:
            pickle.dump(explanation, fe)

        with open("confidence.txt", "wb") as fc:
            pickle.dump(confidence, fc)

        with open("fscore.txt", "wb") as ff:
            pickle.dump(fscore, ff)

        moc = calculate_moc(confidence)
        # print(moc)
        mfscore = calculate_mean_conf(fscore)
        # print(mfscore)

        x = np.arange(10)
        plt.bar(x + 0.2, mfscore, width=0.4, label="f1-score")
        plt.bar(x - 0.2, moc, width=0.4, label="margin")
        plt.xticks(x, causes, rotation=70)
        plt.yticks(np.arange(0, 90, step=10))
        plt.legend()
        plt.show()
    elif number == "2":
        merge_idx = list(np.random.choice(11, 5, replace=False))
        merged_models = []

        for causal_models in all_causal_models:
            merged_model = causal_models[0]
            for i in merge_idx:
                merged_model = m.merge(merged_model, causal_models[i])
            merged_model.print_causal_model()
            merged_models.append(merged_model)

        for merged_model in merged_models:
            merged_model.print_causal_model()

        s_accuracy = [0] * 10
        s_margins = [0] * 10
        s_pred_count = [0] * 10

        # parameter
        theta = 0.2
        num_bins = 500
        threshold_sp = 0.0

        for i in range(10):
            for j in range(11):
                result, margin, pred = s.single_predict(
                    warehouse, all_causal_models[i][j], i, j, causes
                )
                s_accuracy[i] += result
                s_pred_count[pred] += 1
                if result == 1:
                    s_margins[i] += margin

        s_recall = np.array(s_accuracy) / np.array(s_pred_count)
        s_precision = np.array(s_accuracy) / 11
        s_avg_margin = np.array(s_margins) / 11

        s_f1 = 2 * s_precision * s_recall / (s_precision + s_recall)
        print(s_precision, s_recall)
        print(s_f1)

        # parameter
        theta = 0.05
        num_bins = 500
        threshold_sp = 0.0

        accuracy = [0] * 10
        margins = [0] * 10
        pred_count = [0] * 10
        for i in range(10):
            for j in merge_idx:
                result, margin, pred = m.merged_predict(
                    warehouse, all_causal_models[i][j], merge_idx, i, causes
                )
                accuracy[i] += result
                pred_count[pred] += 1
                if result == 1:
                    margins[i] += margin

        recall = np.array(accuracy) / np.array(pred_count)
        precision = np.array(accuracy) / (len(merge_idx))
        avg_margin = np.array(margins) / (len(merge_idx))

        f1 = 2 * precision * recall / (precision + recall)
        # print(precision, recall)
        # print(f1)

        diff_margin = avg_margin - s_avg_margin
        # print(diff_margin)

        x = np.arange(10)

        plt.bar(x - 0.2, s_avg_margin, width=0.4, label="single causal model")
        plt.bar(x + 0.2, avg_margin, width=0.4, label="merged causal model")
        plt.xticks(x, causes, rotation=70)
        plt.yticks(range(0, 90, step=10))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
