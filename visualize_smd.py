import numpy as np
import tqdm
from matplotlib import pyplot as plt

label_file_path = "/root/dbsherlock/data/SMD/SMD_test_label.npy"
data_file_path = "/root/dbsherlock/data/SMD/SMD_test.npy"

label = np.load(label_file_path)
data = np.load(data_file_path)

stop = 1

start_idx = 10000
end_idx = start_idx + 1000
x = range(start_idx, end_idx)
# Find abnormal regions
abnormal_regions = [i for i in x if label[i] != 0]
for i in tqdm.tqdm(range(data.shape[1])):
    y = data[start_idx:end_idx, i]
    # plot
    plt.plot(x, y, color="blue")
    for abnormal_idx in abnormal_regions:
        plt.plot(abnormal_idx, y[abnormal_idx - start_idx], color="red", marker="o")
    plt.savefig(f"{i}.png")
    plt.clf()
