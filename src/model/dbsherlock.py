from typing import *

import numpy as np

from src.data.anomaly_data import AnomalyData
from src.model.partition import Partition, Label, Normal, Abnormal


class DBSherlock:
    def __init__(self, num_discrete: int = 500):
        self.num_discrete = num_discrete
        self.abnormal_multiplier = 10

    def expand_normal_region(self) -> List[int]:
        pass

    def generate_causal_model(self, data: AnomalyData) -> None:
        pass

    def create_partitions(self, data: AnomalyData) -> List[Partition]:
        """Create partitions for each attribute"""
        # Get stats: Max, min, range, and partition size
        max_value = np.argmax(data, axis=0)
        min_value = np.argmin(data, axis=0)
        value_range = max_value - min_value
        partition_size = value_range // self.num_discrete

        paritions: List[Partition] = []
        for idx in range(self.num_discrete):
            # Decide the range of the partition
            partition_start_value = min_value + idx * partition_size
            partition_end_value = min_value + (idx + 1) * partition_size
            # Add the partition
            paritions.append(
                Partition(min=partition_start_value, max=partition_end_value)
            )

        return paritions

    def label_parition(
        self,
        data: np.ndarray,
        partitions: List[Partition],
        normal_regions: List[int],
        abnormal_regions: List[int],
    ) -> List[Partition]:
        """data.shape: (time_steps)"""
        for partition in partitions:
            # Get the time steps of values that belong to this partition
            satisfying_value_idx = [
                idx
                for idx, value in enumerate(data)
                if partition.is_value_in_range(value)
            ]
            # Check if any of the data in the partition is abnormal
            has_normal_values = satisfying_value_idx and any(
                idx in normal_regions for idx in satisfying_value_idx
            )
            has_abnormal_values = satisfying_value_idx and any(
                idx in abnormal_regions for idx in satisfying_value_idx
            )
            # If conflicting labels, label the partition as empty
            if has_normal_values == has_abnormal_values:
                partition.is_empty = True
            else:
                # If no conflicting labels, label the partition
                if has_normal_values:
                    partition.is_normal = True
                else:
                    partition.is_abnormal = True

        return partitions

    def filter_partitions(self, partitions: List[Partition]) -> List[Partition]:
        """Filter out empty partitions"""
        indices_to_filter = []
        for idx in range((len(partitions) - 1)):
            if not partitions[idx].is_empty:
                # Check if the adjacent partitions, which are not empty, has different label
                for adj_idx in range(idx + 1, len(partitions)):
                    if not partitions[adj_idx].is_empty:
                        if partitions[idx].label != partitions[adj_idx].label:
                            indices_to_filter.append(idx)
                            indices_to_filter.append(adj_idx)
                            break
        # Remove duplicates
        indices_to_filter = list(set(indices_to_filter))

        # Count the number of Normal and Abnormal partitions
        num_normal = sum([1 for p in partitions if p.is_normal])
        num_abnormal = sum([1 for p in partitions if p.is_abnormal])
        # Filter (i.e., empty the label) the partitions
        for idx in indices_to_filter:
            # Prevent emptying if there are no more Normal or Abnormal partitions
            if partitions[idx].is_normal and num_normal > 1:
                partitions[idx].is_empty = True
                num_normal -= 1
            elif partitions[idx].is_abnormal and num_abnormal > 1:
                partitions[idx].is_empty = True
                num_abnormal -= 1

        return partitions

    def fill_partition_labels(self, partitions: List[Partition]) -> List[Partition]:
        def calculate_distance_to_nearest_label(start_idx: int, label: Label) -> float:
            """Calculate the distance to the nearest label. Here, the distance is defined
            as the number of partitions between the start_idx and the nearest label.
            """
            distance_to_nearest_label = float("inf")
            # Search forward
            for adj_idx in range(start_idx + 1, len(partitions)):
                if partitions[adj_idx].label == label:
                    distance_to_nearest_label = adj_idx - start_idx
                    break
            # Search backward
            for adj_idx in range(start_idx - 1, -1, -1):
                if partitions[adj_idx].label == label:
                    distance_to_nearest_label = min(
                        distance_to_nearest_label, start_idx - adj_idx
                    )
                    break
            return distance_to_nearest_label

        for idx, partition in enumerate(partitions):
            if partition.is_empty:
                # Calculate distance to the nearest normal and abnormal partition
                distance_to_normal = calculate_distance_to_nearest_label(
                    start_idx=idx, label=Normal()
                )
                distance_to_abnormal = calculate_distance_to_nearest_label(
                    start_idx=idx, label=Abnormal()
                )
                # Label the partition
                if distance_to_normal < distance_to_abnormal * self.abnormal_multiplier:
                    partition.is_normal = True
                else:
                    partition.is_abnormal = True
        return partitions

    def extract_predicate(self, partitions: List[Partition]) -> None:
        pass
