import os
from typing import *

from matplotlib import pyplot as plt

from src.data.anomaly_data import AnomalyData


def plot_data(
    data: AnomalyData,
    cause: str,
    data_id: Optional[int] = None,
    path: Optional[str] = None,
) -> None:
    """Plot data"""
    # Create directory if not exists
    if path:
        os.makedirs(path, exist_ok=True)
    # Plot for each feature
    for att_idx, attribute in enumerate(data.attributes):
        title = f"{cause}-{attribute}-{data_id}".replace(" ", "_")
        plt.title(title)
        x = range(data.values_as_np[:, att_idx].shape[0])
        y = data.values_as_np[:, att_idx]
        # Plot all points in blue
        plt.plot(x, y, color="blue")

        # Highlight points in abnormal regions in red
        for abnormal_idx in data.abnormal_regions:
            plt.plot(x[abnormal_idx], y[abnormal_idx], color="red", marker="o")

        if path:
            plt.savefig(os.path.join(path, f"{title}.png"))
        else:
            plt.show()
        plt.clf()


def plot_performance(
    anomaly_types: List[str],
    confidences: List[float],
    precisions: List[float],
) -> None:
    """Plot performance"""
    plt.title("Confidence and precision for each anomaly type")
    plt.xlabel("Anomaly type")
    plt.ylabel("Confidence and precision")
    plt.xticks(rotation=45)
    plt.bar(anomaly_types, confidences, color="blue")
    plt.bar(anomaly_types, precisions, color="red")
    plt.legend(["Confidence", "Precision"])
    plt.show()
    plt.clf()
