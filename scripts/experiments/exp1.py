import argparse
import logging
import os
from typing import *

import tqdm
import hkkang_utils.file as file_utils

from src.data.anomaly_data import AnomalyData
from src.model.dbsherlock import DBSherlock

logger = logging.getLogger("Exp1")


def main(
    data_path: str,
    output_dir: str,
    total_case_num: int = 10,
    num_sample_per_case: int = 11,
    num_train_samples: int = 1,
) -> None:
    # Check arguments
    assert (
        num_train_samples <= num_sample_per_case
    ), f"Number of train samples should be less than number of samples per case"

    # Load data
    data_in_json = file_utils.read_json_file(data_path)
    anomaly_data_list = [AnomalyData.from_dict(data=d) for d in data_in_json]

    # Check number of data
    assert (
        len(anomaly_data_list) == total_case_num * num_sample_per_case
    ), f"Number of data is less than {total_case_num}"

    # Create causal model
    dbsherlock = DBSherlock()

    # Perform k-fold cross validation (But, for each case train with only one sample and test with the rest)
    for sample_idx in range(num_sample_per_case):
        for case_idx in range(total_case_num):
            training_data = anomaly_data_list[
                case_idx * num_sample_per_case + sample_idx
            ]
            indices = [idx for idx in range(num_sample_per_case) if idx != sample_idx]
            testing_data_list = [
                anomaly_data_list[case_idx * num_sample_per_case + idx]
                for idx in indices
            ]
            # Create causal model
            causal_model = dbsherlock.create_causal_model(data=training_data)
            # Validate confidence
            confidences = []
            for testing_data in testing_data_list:
                confidence = dbsherlock.compute_confidence(
                    causal_model=causal_model, data=testing_data
                )
                confidences.append(confidence)

    raise NotImplementedError
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Conduct experiment 1")
    parser.add_argument(
        "--data",
        type=str,
        help="Path for experimental data",
        default="data/converted_dataset/tpcc_500w_test.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory to save experimental results",
        default="results/exp1/",
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
    data_path = args.data
    output_dir = args.out_dir

    main(data_path=data_path, output_dir=output_dir)
    logger.info("Done!")
