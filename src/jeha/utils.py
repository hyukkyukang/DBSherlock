import statistics as stat


def calculate_mean_conf(confidence):
    num_case = len(confidence)
    mean_conf = [[0 for _ in range(num_case)] for _ in range(num_case)]
    for i in range(num_case):
        for j in range(num_case):
            conf = confidence[i][j]
            mean_conf[i][j] = stat.mean(conf)

    final_conf = [0 for _ in range(num_case)]
    for i in range(num_case):
        final_conf[i] = mean_conf[i][i]

    return final_conf


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

    final_pre = [0 for _ in range(num_case)]
    final_rec = [0 for _ in range(num_case)]
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
