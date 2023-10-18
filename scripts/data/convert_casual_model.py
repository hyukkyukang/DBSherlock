import scipy
import tqdm
import hkkang_utils.file as file_utils

inf = 999999
ninf = -999999


def inf_to_float(x):
    if x > inf:
        return inf
    elif x < ninf:
        return ninf
    else:
        return x


for i in tqdm.tqdm(range(1, 11)):
    for j in range(1, 12):
        input_path = f"/root/dbsherlock/causal_models/cause{i}-{j}.mat"
        data = scipy.io.loadmat(input_path)
        stop = 1

        tmp = {"cause": data["model"]["cause"][0][0].item()}
        for datum in data["model"][0][0][0]:
            att_name = datum[0].item()
            predicates = datum[2].tolist()
            filtered_predicates = []
            for predicate in predicates:
                value0 = inf_to_float(predicate[0])
                value1 = inf_to_float(predicate[1])
                filtered_predicates.append([value0, value1])
            tmp[att_name] = filtered_predicates

        file_utils.write_json_file(tmp, input_path.replace(".mat", ".json"))
