import csv
import math
import pickle

import numpy as np
from numpy.core.fromnumeric import sort
from numpy.lib.function_base import average

NUMERIC = 0
CATEGORICAL = 1
inf = math.inf


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


class predicate:
    def __init__(
        self, predicate_string, attr_num, attr_name, diff, type, lower, upper, c=0
    ):
        self.attr_num = attr_num
        self.attr_name = attr_name
        self.predicate_string = predicate_string
        self.type = type  # numeric = 0 or categorical = 1
        self.lower = lower
        self.upper = upper
        self.diff = diff
        self.c = c

    def get_pred_info(self):
        return self.attr_num, self.lower, self.upper, self.attr_name

    def print_pred(self):
        # print("attr_num is {}".format(self.attr_num))
        if self.type == NUMERIC:
            print(
                "{} {} | diff = {:.4f}".format(
                    self.attr_num + 3, self.predicate_string, self.diff
                )
            )

    def get_diff(self):
        return self.diff


def load_data(warehouse, i, j):
    file_path = f"data/converted_data_{warehouse}/test_datasets/data/{i}_{j}.csv"
    with open(file_path, "r") as f:
        data = list(csv.reader(f, delimiter=","))
    temp = np.array(data, dtype=float)

    timestamp = np.array(temp[:, 0] - 1, dtype=int)
    latency = temp[:, 1]
    temp = temp[:, 2:]
    _, num_attr = temp.shape

    with open(
        "converted_data_"
        + warehouse
        + "/test_datasets/field_names/{}_{}.csv".format(i, j),
        "r",
    ) as f:
        data = list(csv.reader(f, delimiter=","))
    attr_name = data[0][2:]

    with open(
        "converted_data_" + warehouse + "/abnormal_regions/{}_{}.csv".format(i, j), "r"
    ) as f:
        data = list(csv.reader(f, delimiter=","))

    ab_index = np.array(data[0], dtype=int) - 1

    with open(
        "converted_data_" + warehouse + "/normal_regions/{}_{}.csv".format(i, j), "r"
    ) as f:
        data = list(csv.reader(f, delimiter=","))

    if len(data) == 0:
        n_index = np.delete(timestamp, ab_index)
        for i in n_index:
            # print(latency[i])
            if latency[i] <= 0:
                n_index = list(n_index)
                n_index.remove(i)
                n_index = np.array(n_index, dtype=int)
        #
    else:
        n_index = np.array(data[0], dtype=int) - 1

    n = [0 for i in range(num_attr)]  # normal data
    ab = [0 for i in range(num_attr)]  # abnormal data
    d = [0 for i in range(num_attr)]

    for i in range(num_attr):
        n[i] = temp[:, i][n_index]
        ab[i] = temp[:, i][ab_index]
        d_index = np.concatenate((n_index, ab_index))
        sort(d_index)
        d[i] = temp[:, i]

    return num_attr, attr_name, n, ab, d, n_index, ab_index, timestamp


def partition_label(hist_a, hist_n):
    NORMAL = 1
    ABNORMAL = 2
    EMPTY = 0
    n = len(hist_a)
    ret = [0 for _ in range(n)]
    for i in range(n):
        if hist_a[i] == 0 and hist_n[i] != 0:
            # if hist_a[i] < hist_n[i]:
            ret[i] = NORMAL
        elif hist_a[i] != 0 and hist_n[i] == 0:
            # elif hist_a[i] > hist_n[i]:
            ret[i] = ABNORMAL
        else:
            ret[i] = EMPTY
    return ret


def compare(arr, boundary, operator):
    l = len(arr)
    # result = [0 for _ in range(l)]
    b = [boundary for _ in range(l)]
    if boundary > 7e9:
        epsilon = 1e-6
    else:
        epsilon = 1e-7

    is_big = [(arr > boundary) for arr, boundary in zip(arr, b)]
    is_same = [abs(arr - boundary) < epsilon for arr, boundary in zip(arr, b)]

    if operator == ">=":
        is_big = [(arr > boundary) for arr, boundary in zip(arr, b)]
        is_same = [abs(arr - boundary) < epsilon for arr, boundary in zip(arr, b)]
        result = [ai or bi for ai, bi in zip(is_big, is_same)]
        return result
    elif operator == "<":
        is_diff = [abs(arr - boundary) > epsilon for arr, boundary in zip(arr, b)]
        is_small = [(arr < boundary) for arr, boundary in zip(arr, b)]
        result = [ai and bi for ai, bi in zip(is_small, is_diff)]

        return result


def predicate_generation(warehouse, r, c, num_bins, theta, threshold_sp):
    num_attr, attr_name, n, ab, d, n_index, ab_index, timestamp = load_data(
        warehouse, r, c
    )
    # Creating a Partition Space
    NORMAL = 1
    ABNORMAL = 2
    EMPTY = 0

    boundaries = [[] for i in range(num_attr)]
    label = [[] for i in range(num_attr)]
    max_value = []
    min_value = []

    # Partition Labeling

    skip_attr = []
    normalized_normal_avg = [0 for _ in range(num_attr)]
    normalized_abnormal_avg = [0 for _ in range(num_attr)]

    with open("single/label/{}_{}.txt".format(r, c), "rb") as fl:
        label = pickle.load(fl)

    for i in range(num_attr):
        max_value.append(d[i].max())
        min_value.append(d[i].min())
        attr_range = max_value[i] - min_value[i]
        bin_size = (max_value[i] - min_value[i]) / num_bins

        boundaries[i] = np.linspace(
            min_value[i], max_value[i], num_bins, endpoint=False
        )
        boundaries[i] = np.append(boundaries[i], max_value[-1])

        for k in range(len(boundaries[i])):
            boundaries[i][k] = np.round(boundaries[i][k], 10)
        boundary_count = len(boundaries[i])
        if bin_size == 0:
            continue
        # print('{:.4f} {:.4f} {:.4f} {:.4f} {}'.format(max_value[-1], min_value[-1], attr_range, bin_size, len(boundaries[i])))

    # with open('single/label/{}_{}.txt'.format(r,c), 'wb') as fl:
    #    pickle.dump(label, fl)

    for i in range(num_attr):
        attr_range = max_value[i] - min_value[i]
        bin_size = (max_value[i] - min_value[i]) / num_bins

        normalized_normal_sum = 0
        normalized_normal_count = 0
        normalized_abnormal_sum = 0
        normalized_abnormal_count = 0

        for j in range(len(label[i])):
            if label[i][j] == NORMAL:
                normalized_normal_sum += (boundaries[i][j] - min_value[i]) / attr_range
                normalized_normal_count += 1  # hist_n[i][j]
            elif label[i][j] == ABNORMAL:
                normalized_abnormal_sum += (
                    boundaries[i][j] - min_value[i]
                ) / attr_range
                normalized_abnormal_count += 1  # hist_a[i][j]

        if normalized_normal_count == 0:
            normalized_normal_avg[i] = math.nan
        else:
            normalized_normal_avg[i] = normalized_normal_sum / normalized_normal_count

        if normalized_abnormal_count == 0:
            normalized_abnormal_avg[i] = math.nan
        else:
            normalized_abnormal_avg[i] = (
                normalized_abnormal_sum / normalized_abnormal_count
            )

    normalized_normal_avg = np.array(normalized_normal_avg)
    normalized_abnormal_avg = np.array(normalized_abnormal_avg)

    # Partition Filtering

    for i in range(num_attr):
        f_list = []

        normal_count = label[i].count(NORMAL)
        abnormal_count = label[i].count(ABNORMAL)

        for j in range(len(label[i]) - 1):
            if label[i][j] == EMPTY:
                continue
            for k in range(j + 1, len(label[i])):
                if label[i][k] == EMPTY:
                    continue
                elif label[i][j] != label[i][k]:
                    if j not in f_list:
                        f_list.append(j)
                    if k not in f_list:
                        f_list.append(k)
                    break
                else:
                    break

        for j in f_list:
            if normal_count > 1 and label[i][j] == NORMAL:
                label[i][j] = EMPTY
            elif abnormal_count > 1 and label[i][j] == ABNORMAL:
                label[i][j] = EMPTY

        normal_count = label[i].count(NORMAL)
        abnormal_count = label[i].count(ABNORMAL)
        if normal_count == 0 and abnormal_count > 0:
            normal_mean = average(n[i])
            for j in range(len(boundaries[i])):
                if j == len(boundaries[i]) - 1:
                    if normal_mean >= boundaries[i][j]:
                        label[i][j] = NORMAL
                        break
                else:
                    if (
                        normal_mean >= boundaries[i][j]
                        and normal_mean < boundaries[i][j + 1]
                    ):
                        label[i][j] = NORMAL
                        break

    # filling the gap
    abnormal_multiplier = 10
    recheck_attr = []
    empty = []
    non_empty = []
    for i in range(num_attr):
        if i in skip_attr:
            continue

        normal_count = label[i].count(NORMAL)
        abnormal_count = label[i].count(ABNORMAL)

        mark = [0 for _ in range(len(label[i]))]

        for j in range(len(label[i])):
            if label[i][j] == EMPTY:
                dist_normal = num_bins * 2 * abnormal_multiplier
                dist_abnormal = num_bins * 2 * abnormal_multiplier
            else:
                continue
            k = j - 1
            while k >= 0:
                if label[i][k] == NORMAL:
                    if dist_normal > abs(k - j):
                        dist_normal = abs(k - j)
                    break
                if label[i][k] == ABNORMAL:
                    if dist_abnormal > abs(k - j) * abnormal_multiplier:
                        dist_abnormal = abs(k - j) * abnormal_multiplier
                    break
                k -= 1

            k = j + 1
            while k < len(label[i]):
                if label[i][k] == NORMAL:
                    if dist_normal > abs(k - j):
                        dist_normal = abs(k - j)
                    break
                if label[i][k] == ABNORMAL:
                    if dist_abnormal > abs(k - j) * abnormal_multiplier:
                        dist_abnormal = abs(k - j) * abnormal_multiplier
                    break
                k += 1

            if dist_abnormal > dist_normal:
                mark[j] = NORMAL
            elif dist_abnormal <= dist_normal:
                mark[j] = ABNORMAL

        for j in range(len(label[i])):
            if label[i][j] == 0:
                label[i][j] = mark[j]
        # print(*label[i])

    # Extracting Predicates from Partitions
    predicates = {}
    diff = np.abs(normalized_normal_avg - normalized_abnormal_avg)
    diff = np.nan_to_num(diff, nan=0)

    filtered_attr = []
    for i in range(num_attr):
        if (
            len(label[i]) == 0 or label[i].count(ABNORMAL) == 0
        ):  # or label[i].count(NORMAL)==0:
            continue
        if attr_name[i].startswith("dbmsCum"):
            continue

        if diff[i] < theta:
            continue

        lower = -inf
        upper = inf
        predicate_count = 0
        predicate_string = ""

        for j in range(len(label[i]) - 1):
            if j == 0 and label[i][j] == ABNORMAL:
                predicate_count += 1
            if (
                label[i][j] != ABNORMAL and label[i][j + 1] == ABNORMAL
            ):  # attr < boundary[i][j+1]
                if len(predicate_string) != 0:
                    predicate_string += " OR  > {:.6f}".format(boundaries[i][j + 1])
                else:
                    predicate_string = " > {:.6f}".format(boundaries[i][j + 1])

                lower = boundaries[i][j + 1]
                predicate_count += 1
            elif label[i][j] == ABNORMAL and label[i][j + 1] != ABNORMAL:
                if len(predicate_string) != 0:
                    predicate_string += "and "
                predicate_string += " < {:.6f}".format(boundaries[i][j + 1])

                upper = boundaries[i][j + 1]
                predicate_count += 1

        if len(predicate_string) != 0:
            predicate_string = attr_name[i] + predicate_string
        if predicate_count > 0 and predicate_count <= 2:
            predicates[attr_name[i]] = predicate(
                predicate_string, i, attr_name[i], diff[i], NUMERIC, lower, upper
            )
            filtered_attr.append(i)

    temp = [predicates[attr_name[x]].get_diff() for x in filtered_attr]
    temp = np.nan_to_num(temp, nan=0)
    sorted_attr = [x for _, x in sorted(zip(temp, filtered_attr), reverse=True)]
    # sorted(filtered_attr, key=lambda x: temp, reverse=True)
    ret = {}

    for i in sorted_attr:
        if predicates[attr_name[i]].get_diff() > theta:
            # predicates[attr_name[i]].print_pred()
            # predicates[sorted_sp[i][0]].plot(ab_index, n_index, ab, n, timestamp)
            ret[attr_name[i]] = predicates[attr_name[i]]

    return ret


class causal_model:
    def __init__(self, cv, eps):
        self.cv = cv
        self.eps = eps

    def cal_confidence(self, n, ab, d, num_bins, r, c):
        with open("single/label/{}_{}.txt".format(r, c), "rb") as fl:
            labels = pickle.load(fl)

        # r와 c는 출력해서 디버깅할때 사용
        ABNORMAL = 2
        NORMAL = 1
        EMPTY = 0

        covered_abnormal_ratio_average = 0
        covered_normal_ratio_average = 0
        precision_average = 0
        recall_average = 0

        for ep in self.eps.values():
            i, lower, upper, name = ep.get_pred_info()
            label = labels[i]

            max_value = d[i].max()
            min_value = d[i].min()
            bin_size = (max_value - min_value) / (num_bins)
            if bin_size == 0:
                continue

            # print(['conf : {} before boundary'.format(i),t])

            boundary = np.linspace(min_value, max_value, num_bins, endpoint=False)
            boundary = np.append(boundary, max_value)

            for k in range(len(boundary)):
                boundary[k] = np.round(boundary[k], 10)
            boundary_count = len(boundary)

            # print(['conf : {} label'.format(i),t])

            normal_partition_count = 0
            abnormal_partition_count = 0
            covered_partition_count = 0
            covered_normal_count = 0
            for l in range(len(label)):
                if label[l] == ABNORMAL:
                    abnormal_partition_count += 1
                    if lower != -inf and lower > boundary[l]:
                        pass
                    elif (
                        upper != inf
                        and l != len(label) - 1
                        and upper <= boundary[l + 1]
                    ):
                        pass
                    else:
                        covered_partition_count += 1
                elif label[l] == NORMAL:
                    normal_partition_count += 1
                    if lower != -inf and lower > boundary[l]:
                        pass
                    elif (
                        upper != inf
                        and l != len(label) - 1
                        and upper <= boundary[l + 1]
                    ):
                        pass
                    else:
                        covered_normal_count += 1

            if abnormal_partition_count == 0:
                ratio = 0
            else:
                ratio = covered_partition_count / abnormal_partition_count
            covered_abnormal_ratio_average += ratio

            if normal_partition_count == 0:
                ratio = 0
            else:
                ratio = covered_normal_count / normal_partition_count
            covered_normal_ratio_average += ratio

            if covered_normal_count + covered_partition_count == 0:
                ratio = 0
            else:
                ratio = covered_partition_count / (
                    covered_normal_count + covered_partition_count
                )
            precision_average += ratio

            # print(name, covered_partition_count, covered_normal_count, abnormal_partition_count, normal_partition_count)
            # print("{} {} {} {} {} {} {} {}".format(r, c, i+3, name, covered_partition_count, covered_normal_count, abnormal_partition_count, normal_partition_count))

        covered_abnormal_ratio_average = covered_abnormal_ratio_average / len(self.eps)
        covered_normal_ratio_average = covered_normal_ratio_average / len(self.eps)
        precision_average = precision_average / len(self.eps)

        confidence = (
            covered_abnormal_ratio_average - covered_normal_ratio_average
        ) * 100
        precision = precision_average * 100
        recall = covered_abnormal_ratio_average * 100
        if (covered_abnormal_ratio_average + precision_average) == 0:
            f1_score = 0
        else:
            f1_score = (
                2
                * covered_abnormal_ratio_average
                * precision_average
                / (covered_abnormal_ratio_average + precision_average)
                * 100
            )

        return (
            self.cv,
            confidence,
            precision,
            recall,
            f1_score,
            covered_normal_ratio_average * 100,
        )

    def cal_confidence_test(self, n, ab, d, num_bins, r, c, prior_attr):
        with open("single/label/{}_{}.txt".format(r, c), "rb") as fl:
            labels = pickle.load(fl)

        warehouse = "500"
        with open("converted_data_" + warehouse + "/causes.csv", "r") as f:
            data = list(csv.reader(f, delimiter=","))
        causes = data[0]

        # r와 c는 출력해서 디버깅할때 사용
        ABNORMAL = 2
        NORMAL = 1
        EMPTY = 0
        multiplier = 1

        covered_abnormal_ratio_average = 0
        covered_normal_ratio_average = 0
        precision_average = 0
        recall_average = 0

        covered_abnormal_ratio = []
        covered_normal_ratio = []
        sp = []
        mul = []

        for x in range(len(causes)):
            if causes[x] == self.cv:
                case = x
            """         
            if len(prior_attr[case])!=0:
            min_values = min(list(prior_attr[case].values()))
            max_values = max(list(prior_attr[case].values()))
            value_range = max_values - min_values
            if value_range == 0:
                value_range = 1
            scale = 5 """

        for ep in self.eps.values():
            i, lower, upper, name = ep.get_pred_info()
            multiplier = 1

            if len(prior_attr[case]) != 0:
                if i in prior_attr[case]:
                    multiplier = (prior_attr[case][i] - 6) * 3
                    # (prior_attr[case][i]-min_values)*scale/value_range+1

            # if self.cv == causes[5]:
            #    if i == 191 or i == 192:
            #        multiplier = 10

            # if self.cv == causes[7]:
            #    if i == 187 or i == 196:
            #        multiplier = 10

            label = labels[i]

            max_value = d[i].max()
            min_value = d[i].min()
            bin_size = (max_value - min_value) / (num_bins)
            if bin_size == 0:
                continue

            # print(['conf : {} before boundary'.format(i),t])

            boundary = np.linspace(min_value, max_value, num_bins, endpoint=False)
            boundary = np.append(boundary, max_value)

            for k in range(len(boundary)):
                boundary[k] = np.round(boundary[k], 10)
            boundary_count = len(boundary)

            # print(['conf : {} label'.format(i),t])

            normal_partition_count = 0
            abnormal_partition_count = 0
            covered_partition_count = 0
            covered_normal_count = 0
            for l in range(len(label)):
                if label[l] == ABNORMAL:
                    abnormal_partition_count += 1
                    if lower != -inf and lower > boundary[l]:
                        pass
                    elif (
                        upper != inf
                        and l != len(label) - 1
                        and upper <= boundary[l + 1]
                    ):
                        pass
                    else:
                        covered_partition_count += 1
                elif label[l] == NORMAL:
                    normal_partition_count += 1
                    if lower != -inf and lower > boundary[l]:
                        pass
                    elif (
                        upper != inf
                        and l != len(label) - 1
                        and upper <= boundary[l + 1]
                    ):
                        pass
                    else:
                        covered_normal_count += 1

            if abnormal_partition_count == 0:
                ratio = 0
            else:
                ratio = covered_partition_count / abnormal_partition_count
            covered_abnormal_ratio.append(ratio)
            covered_abnormal_ratio_average += ratio

            if normal_partition_count == 0:
                ratio = 0
            else:
                ratio = covered_normal_count / normal_partition_count
            covered_normal_ratio_average += ratio
            covered_normal_ratio.append(ratio)

            if covered_normal_count + covered_partition_count == 0:
                ratio = 0
            else:
                ratio = covered_partition_count / (
                    covered_normal_count + covered_partition_count
                )
            precision_average += ratio

            sp.append(
                (covered_abnormal_ratio[-1] - covered_normal_ratio[-1]) * multiplier
            )
            mul.append(multiplier)
            # if covered_abnormal_ratio[-1]-covered_normal_ratio[-1] > 0.5:
            #    covered_abnormal_ratio[-1]*=multiplier
            #    covered_normal_ratio[-1]*=multiplier

            # print(name, covered_partition_count, covered_normal_count, abnormal_partition_count, normal_partition_count)
            # print("{} {} {} {} {} {} {} {}".format(r, c, i+3, name, covered_partition_count, covered_normal_count, abnormal_partition_count, normal_partition_count))

        # sp = np.array(covered_abnormal_ratio)-np.array(covered_normal_ratio)
        # sp /= np.sum(sp)
        normalizing_factor = np.linalg.norm(np.array(mul)) / np.linalg.norm(
            np.ones_like(mul)
        )
        sp /= normalizing_factor**0.25
        # print("modified version")

        # weight = softmax(sp)
        confidence = np.sum(sp) / len(self.eps) * 100

        covered_abnormal_ratio_average = covered_abnormal_ratio_average / len(self.eps)
        covered_normal_ratio_average = covered_normal_ratio_average / len(self.eps)
        precision_average = precision_average / len(self.eps)

        # confidence = (covered_abnormal_ratio_average-covered_normal_ratio_average)*100
        precision = precision_average * 100
        recall = covered_abnormal_ratio_average * 100
        if (covered_abnormal_ratio_average + precision_average) == 0:
            f1_score = 0
        else:
            f1_score = (
                2
                * covered_abnormal_ratio_average
                * precision_average
                / (covered_abnormal_ratio_average + precision_average)
                * 100
            )

        return (
            self.cv,
            confidence,
            precision,
            recall,
            f1_score,
            covered_normal_ratio_average * 100,
        )

    def print_causal_model(self):
        print("Cause Variable : {}".format(self.cv))
        for k, v in iter(self.eps.items()):
            v.print_pred()

    def get_cv(self):
        return self.cv

    def get_eps(self):
        return self.eps


# precision = precision average/ effective predicate
# recall = average abnormal ratio
