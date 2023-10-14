import argparse
import logging
import os
from typing import *

import hkkang_utils.file as file_utils

from src.data.anomaly_data import AnomalyData
from src.model.dbsherlock import DBSherlock

logger = logging.getLogger("Exp1")


def main(data_path: str, output_dir: str) -> None:
    # Load data
    data_in_json = file_utils.read_json_file(data_path)
    anomaly_data_list = [AnomalyData.from_dict(data=d) for d in data_in_json]
    # Create causal model
    dbsherlock = DBSherlock()
    dbsherlock.create_causal_model(data=anomaly_data_list[0])
    raise NotImplementedError


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
