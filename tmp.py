import json
from src.data.anomaly_data import AnomalyData

data_path = "data/converted_dataset/tpcc_500w_test.json"

# Load the dataset in json format
data_in_json = json.load(open(data_path, "r"))

# Load the dataset in AnomalyData format
data_list = [AnomalyData.from_dict(data=d) for d in data_in_json]

tmp = 1


num_discrete = 500
diff_threshold = 0.2000

for i in range(10):
    for j in range(1, 11):  # use data except for the first one
        pass
