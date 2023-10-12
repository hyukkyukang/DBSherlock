# How to load the dataset in Python
You can load the dataset in json format or in AnomalyData format. The AnomalyData format is a wrapper of the json format, which provides some useful functions.
```python
import json
from src.data.anomaly_data import AnomalyData

data_path = "data/converted_dataset/tpcc_16w_test.json"

# Load the dataset in json format
data_in_json = json.load(open(data_path, 'r'))

# Load the dataset in AnomalyData format
data = AnomalyData.from_dict(data=data_in_json)
```