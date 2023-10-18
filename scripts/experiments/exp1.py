import argparse
import logging
from typing import *

import hkkang_utils.file as file_utils
import tqdm

from src.data.anomaly_data import AnomalyDataset
from src.data.visualize import plot_performance
from src.model.dbsherlock import DBSherlock

logger = logging.getLogger("Exp1")


def main(
    data_path: str,
    output_dir: str,
    num_sample_per_case: int = 11,
    num_train_samples: int = 1,
) -> None:
    # Check arguments
    assert (
        num_train_samples <= num_sample_per_case
    ), f"Number of train samples should be less than number of samples per case"

    # Load data
    data_in_json = file_utils.read_json_file(data_path)
    anomaly_dataset = AnomalyDataset.from_dict(data=data_in_json)

    # Check number of data
    assert (
        len(anomaly_dataset) == len(anomaly_dataset.causes) * num_sample_per_case
    ), f"Number of data is not correct, {len(anomaly_dataset)} vs {len(anomaly_dataset.causes) * num_sample_per_case}"

    # Create causal model
    dbsherlock = DBSherlock()

    # Perform k-fold cross validation (But, for each case train with only one sample and test with the rest)
    confidence_dic = {cause: [] for cause in anomaly_dataset.causes}
    precision_dic = {cause: [] for cause in anomaly_dataset.causes}
    pbar_for_instance = tqdm.tqdm(range(num_sample_per_case))
    for instance_idx in pbar_for_instance:
        pbar_for_instance.set_description(f"Instance: {instance_idx+1}")
        pbar_for_cause = tqdm.tqdm(anomaly_dataset.causes)
        for cause_idx, anomaly_cause in enumerate(pbar_for_cause):
            pbar_for_cause.set_description(f"Cause: {anomaly_cause}")
            training_data = anomaly_dataset[
                cause_idx * num_sample_per_case + instance_idx
            ]
            indices = [idx for idx in range(num_sample_per_case) if idx != instance_idx]
            testing_data_list = [
                anomaly_dataset[cause_idx * num_sample_per_case + idx]
                for idx in indices
            ]
            # Create causal model
            causal_model = dbsherlock.create_causal_model(data=training_data)
            # Compute confidence and precision for each testing data
            confidences: List[float] = []
            precisions: List[float] = []
            for testing_data in testing_data_list:
                confidence, precision = dbsherlock.compute_confidence(
                    causal_model=causal_model, data=testing_data
                )
                confidences.append(confidence)
                precisions.append(precision)
            # Average over all testing data for each anomaly cause
            avg_confidence = sum(confidences) / len(confidences)
            avg_precision = sum(precisions) / len(precisions)
            confidence_dic[anomaly_cause].append(avg_confidence)
            precision_dic[anomaly_cause].append(avg_precision)

    # Summarize results
    for cause, confidences in confidence_dic.items():
        confidence_dic[cause] = sum(confidences) / len(confidences)
    for cause, precisions in precision_dic.items():
        precision_dic[cause] = sum(precisions) / len(precisions)

    # Plot results
    plot_performance(
        anomaly_causes=anomaly_dataset.causes,
        confidences=list(confidence_dic.values()),
        precisions=list(precision_dic.values()),
        path=output_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Conduct experiment 1")
    parser.add_argument(
        "--data",
        type=str,
        help="Path for experimental data",
        default="data/converted_dataset/tpcc_500w_test.json",
    )
    parser.add_argument(
        "--output_dir",
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
    output_dir = args.output_dir

    main(data_path=data_path, output_dir=output_dir)
    logger.info("Done!")
