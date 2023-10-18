import argparse
import logging
from typing import *

import hkkang_utils.file as file_utils
import tqdm

from src.data.anomaly_data import AnomalyData, AnomalyDataset
from src.data.visualize import plot_data

logger = logging.getLogger("DataVisualizer")


def main(data_path: str, output_path: str, plot_all_data: bool = False) -> None:
    # Load data
    data_in_json = file_utils.read_json_file(data_path)
    anomaly_dataset = AnomalyDataset.from_dict(data=data_in_json)

    # Group data by causes
    cause_to_anomaly_data_list = {}
    for cause in anomaly_dataset.causes:
        data_for_cause: List[AnomalyData] = anomaly_dataset.get_data_of_cause(cause)
        cause_to_anomaly_data_list[cause] = data_for_cause

    # Visualize data
    pbar_for_cause_data = tqdm.tqdm(cause_to_anomaly_data_list.items())
    for cause, anomaly_data_list in pbar_for_cause_data:
        # Get output path for each cause
        output_path_for_cause = os.path.join(output_path, cause.replace("/", ""))
        # Create plot
        pbar_for_cause_data.set_description(f"Visualizing cause: {cause}")
        if plot_all_data:
            pbar_for_instance = tqdm.tqdm(anomaly_data_list)
            for data_idx, data in enumerate(
                pbar_for_instance, total=len(anomaly_data_list)
            ):
                pbar_for_instance.set_description(f"Visualizing instance: {data_idx}")
                plot_data(
                    data, cause=cause, data_id=data_idx, path=output_path_for_cause
                )
        else:
            plot_data(
                anomaly_data_list[0], cause=cause, data_id=0, path=output_path_for_cause
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Plot and visualize data")
    parser.add_argument(
        "--data",
        type=str,
        default="data/converted_dataset/tpcc_500w_test.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/visualize_data/",
    )
    parser.add_argument(
        "--plot_all",
        action="store_true",
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
    output_path = args.output
    plot_all = args.plot_all

    main(data_path=data_path, output_path=output_path, plot_all_data=plot_all)

    logger.info("Done!")
