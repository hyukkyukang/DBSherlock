import argparse
import logging
from typing import *

import hkkang_utils.file as file_utils
import numpy as np
from matplotlib import pyplot as plt

from src.data.anomaly_data import AnomalyData

logger = logging.getLogger("DataConverter")


def main(input_path: str) -> None:
    # Load data
    data_in_json = file_utils.read_json_file(input_path)
    anomaly_data_list = [AnomalyData.from_dict(data=d) for d in data_in_json]

    # Group data by causes
    cause_to_anomaly_data_list = {}
    for anomaly_data in anomaly_data_list:
        cause = anomaly_data.cause
        if cause not in cause_to_anomaly_data_list:
            cause_to_anomaly_data_list[cause] = []
        cause_to_anomaly_data_list[cause].append(anomaly_data)

    # Visualize data
    for cause, anomaly_data_list in cause_to_anomaly_data_list.items():
        logger.info(f"cause: {cause}, # of anomaly data: {len(anomaly_data_list)}")
        for data_idx, data in enumerate(anomaly_data_list):
            # Plot for each feature
            for att_idx, attribute in enumerate(data.attributes):
                logger.info(f"att_idx: {att_idx}, att: {attribute}")
                x = range(data.values_as_np[:, att_idx].shape[0])
                y = data.values_as_np[:, att_idx]
                plt.plot(x, y)
                plt.show()

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .mat to .csv")
    parser.add_argument(
        "--input",
        type=str,
        default="data/converted_dataset/tpcc_500w_test.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    # Arguments
    input_path = args.input

    main(input_path=input_path)
    logger.info("Done!")
