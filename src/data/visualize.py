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
            os.makedirs(path, exist_ok=True)
            # Replace "/" in title to avoid creating subdirectories
            title = title.replace("/", "")
            plt.savefig(os.path.join(path, f"{title}.png"))
        else:
            plt.show()
        plt.clf()


def plot_performance(
    anomaly_causes: List[str],
    confidences: List[float],
    precisions: List[float],
    path: Optional[str] = None,
) -> None:
    """Plot performance"""
    plt.title("Confidence and precision for each anomaly cause")
    plt.xlabel("Anomaly cause")
    plt.ylabel("Confidence and precision")

    bar_width = 0.35
    r1 = range(len(anomaly_causes))
    r2 = [x + bar_width for x in r1]

    plt.bar(
        r1,
        confidences,
        color="blue",
        width=bar_width,
        edgecolor="grey",
        label="Confidence",
    )
    plt.bar(
        r2,
        precisions,
        color="red",
        width=bar_width,
        edgecolor="grey",
        label="Precision",
    )

    plt.xlabel("Anomaly cause", fontweight="bold")
    plt.xticks(
        [r + bar_width for r in range(len(anomaly_causes))], anomaly_causes, rotation=45
    )
    plt.legend()
    plt.tight_layout()

    if path:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "performance.png"))
    else:
        plt.show()
    plt.clf()
