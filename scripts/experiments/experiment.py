import argparse
import logging
from typing import *

import hkkang_utils.file as file_utils
import tqdm

from src.data.anomaly_data import AnomalyData, AnomalyDataset
from src.data.visualize import plot_performance
from src.model.dbsherlock import DBSherlock

logger = logging.getLogger("Experiment")


def split_dataset(
    data: AnomalyDataset, cause: str, target_idx: int, exp_id: int
) -> Tuple[List[AnomalyData], List[AnomalyData]]:
    if exp_id == 1:
        # Use one training data and the rest for testing
        target_data = data.get_data_of_cause(cause=cause)
        training_data = [target_data[target_idx]]
        testing_data = [
            data for idx, data in enumerate(target_data) if idx != target_idx
        ]
    elif exp_id == 2:
        # Use one testing data and the rest for training
        target_data = data.get_data_of_cause(cause=cause)
        testing_data = [target_data[target_idx]]
        training_data = [
            data for idx, data in enumerate(target_data) if idx != target_idx
        ]
    else:
        ValueError(f"Invalid exp_id: {exp_id}")
    return training_data, testing_data


def main(
    exp_id: int,
    data_path: str,
    output_dir: str,
    num_sample_per_case: int = 11,
) -> None:
    # Load data
    data_in_json = file_utils.read_json_file(data_path)
    anomaly_dataset = AnomalyDataset.from_dict(data=data_in_json)

    # Check number of data
    assert (
        len(anomaly_dataset) == len(anomaly_dataset.causes) * num_sample_per_case
    ), f"Number of data is not correct, {len(anomaly_dataset)} vs {len(anomaly_dataset.causes) * num_sample_per_case}"

    # Create DBSherlockmodel
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

            # Get training and testing data
            training_dataset, testing_dataset = split_dataset(
                data=anomaly_dataset,
                cause=anomaly_cause,
                target_idx=instance_idx,
                exp_id=exp_id,
            )

            # Create and merge causal model
            causal_models = []
            for training_data in training_dataset:
                causal_models.append(dbsherlock.create_causal_model(data=training_data))
            merged_causal_model = sum(causal_models)

            # Compute confidence and precision for each testing data
            confidences: List[float] = []
            precisions: List[float] = []
            for testing_data in testing_dataset:
                confidence, precision = dbsherlock.compute_confidence(
                    causal_model=merged_causal_model, data=testing_data
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
        "--exp_id",
        type=str,
        help="Experiment ID",
        choices=["1", "2"],
        default="1",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path for experimental data",
        default="data/converted_dataset/tpcc_16w_test.json",
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
    exp_id = int(args.exp_id)
    data_path = args.data
    output_dir = args.output_dir
    logger.info(f"Running experiment 2 with data: {data_path}")
    main(exp_id=exp_id, data_path=data_path, output_dir=output_dir)
    logger.info(f"Done! Results are saved in {output_dir}")
