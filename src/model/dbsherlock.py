import copy
from typing import *

import hkkang_utils.list as list_utils
import numpy as np

from src.data.anomaly_data import AnomalyData
from src.model.causal_model import CausalModel, Predicate
from src.model.partition import Abnormal, Label, Normal, Partition


class DBSherlock:
    def __init__(
        self,
        num_discrete: int = 500,
        abnormal_multipler: int = 10,
        normalized_difference_threshold: float = 0.2,
        domain_knowledge: Optional[str] = None,
    ):
        self.num_discrete = num_discrete
        self.abnormal_multiplier = abnormal_multipler
        self.normalized_difference_threshold = normalized_difference_threshold
        self.domain_knowledge = domain_knowledge

    def expand_normal_region(self) -> List[int]:
        raise NotImplementedError

    def create_partitions(self, data: AnomalyData) -> List[List[Partition]]:
        """Create partitions for each attribute"""
        # Get stats: Max, min, range, and partition size
        paritions_by_attr: List[List[Partition]] = []
        for att_idx, attribute in enumerate(data.valid_attributes):
            values = data.valid_values_as_np[:, att_idx]
            max_value = max(values)
            min_value = min(values)
            value_range = max_value - min_value
            if value_range == 0:  # Handle case where all values are the same
                paritions_by_attr.append([])
                continue
            partition_size = value_range / self.num_discrete

            paritions: List[Partition] = []
            for idx in range(self.num_discrete):
                # Decide the range of the partition
                partition_start_value = min_value + idx * partition_size
                partition_end_value = min_value + (idx + 1) * partition_size
                # Add the partition
                paritions.append(
                    Partition(
                        attribute=attribute,
                        max=partition_end_value,
                        min=partition_start_value,
                    )
                )

            # Add data to the partitions
            for value in values:
                for partition in paritions:
                    if partition.is_value_in_range(value):
                        partition.values.append(value)
                        break
            paritions_by_attr.append(paritions)

        return paritions_by_attr

    def label_parition(
        self,
        values: np.ndarray,
        partitions: List[Partition],
        normal_regions: List[int],
        abnormal_regions: List[int],
    ) -> List[Partition]:
        """values.shape: (time_steps)"""
        for partition in partitions:
            # Get the time steps of values that belong to this partition
            satisfying_value_idx = [
                idx
                for idx, value in enumerate(values.tolist())
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

    def is_to_extract_predicates(self, partitions: List[Partition]) -> bool:
        """
        This method checks if the attribute is to be used for extracting predicates.
        This should be called on partitions before filtering and filling the partitions
        """
        if len(partitions) == 0:
            return False
        # Calculate the max, min, and range of all values
        all_values = list_utils.do_flatten_list([p.values for p in partitions])
        max_value, min_value = max(all_values), min(all_values)
        value_range = max_value - min_value

        # Calculate average normalized values of normal and abnormal partitions
        normalized_normal_sum = sum(
            [(p.min - min_value) / value_range for p in partitions if p.is_normal]
        )
        normal_cnt = sum([1 for p in partitions if p.is_normal])
        normalized_abnormal_sum = sum(
            [(p.min - min_value) / value_range for p in partitions if p.is_abnormal]
        )
        abnormal_cnt = sum([1 for p in partitions if p.is_abnormal])

        # Handle case where there are no abnormal partitions
        if abnormal_cnt == 0 or normal_cnt == 0:
            return False

        # calculate average normalized values
        avg_normalized_normal = normalized_normal_sum / normal_cnt
        avg_normalized_abnormal = normalized_abnormal_sum / abnormal_cnt

        # Check if the difference between the average normalized values of normal and abnormal is greater than the threshold
        difference = abs(avg_normalized_normal - avg_normalized_abnormal)
        return difference > self.normalized_difference_threshold

    def filter_partitions(self, partitions: List[Partition]) -> List[Partition]:
        """Filtering: For each partition, convert to empty label if the adjacent partitions have different labels"""
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

    def extract_predicate(self, partitions: List[Partition]) -> List[Predicate]:
        attribute = partitions[0].attribute
        predicates = []
        for idx in range(len(partitions) - 1):
            current_partition = partitions[idx]
            next_partition = partitions[idx + 1]

            # Make sure to start the range if the first partition is abnormal
            if idx == 0 and current_partition.is_abnormal:
                predicates += [(">", current_partition.min)]
            # End the range
            if current_partition.is_abnormal and next_partition.is_normal:
                predicates += [("<", current_partition.max)]
            # Start the range
            if current_partition.is_normal and next_partition.is_abnormal:
                predicates += [(">", current_partition.min)]

        # Format predicates as DNF
        predicate_as_dnf: List[Predicate] = []
        for idx in range(0, len(predicates), 2):
            if idx == len(predicates) - 1:
                assert (
                    predicates[idx][0] == ">"
                ), f"Invalid predicate: {predicates[idx]}"
                # Single literal
                predicate_as_dnf += [
                    Predicate(
                        attribute=attribute,
                        operator1=predicates[idx][0],
                        operand1=predicates[idx][1],
                    )
                ]
            else:
                # Multiple literals as conjunction
                assert (
                    predicates[idx][0] == ">" and predicates[idx + 1][0] == "<"
                ), f"Invalid predicate: {predicates[idx]} and {predicates[idx + 1]}"
                predicate_as_dnf += [
                    Predicate(
                        attribute=attribute,
                        operator1=predicates[idx][0],
                        operand1=predicates[idx][1],
                        operator2=predicates[idx + 1][0],
                        operand2=predicates[idx + 1][1],
                    )
                ]
        return predicate_as_dnf

    def create_causal_model(self, data: AnomalyData) -> CausalModel:
        # Create partitions
        partitions_by_attr: List[List[Partition]] = self.create_partitions(data)
        # Label partitions
        partitions_labeled: List[List[Partition]] = []
        for idx, partitions in enumerate(partitions_by_attr):
            labeled_partitions: List[Partition] = self.label_parition(
                values=data.valid_values_as_np[:, idx],
                partitions=partitions,
                normal_regions=data.valid_normal_regions,
                abnormal_regions=data.valid_abnormal_regions,
            )
            partitions_labeled.append(labeled_partitions)

        # Get only the partitions to be used for extracting predicates
        partitions_to_use: List[List[Partition]] = list(
            filter(self.is_to_extract_predicates, partitions_labeled)
        )
        # Filter partitions
        partitions_copied = copy.deepcopy(partitions_to_use)
        filtered_partitions: List[List[Partition]] = list(
            map(self.filter_partitions, partitions_copied)
        )

        # Fill partition labels
        filled_partitions: List[List[Partition]] = list(
            map(self.fill_partition_labels, filtered_partitions)
        )

        # Extract predicates
        extracted_predicates: List[List[Predicate]] = list(
            map(self.extract_predicate, filled_partitions)
        )

        # Filter attributes with only one predicate
        filtered_predicates: List[List[Predicate]] = [
            predicates for predicates in extracted_predicates if len(predicates) == 1
        ]

        # Create causal model
        causal_model = CausalModel(
            cause=data.cause,
            predicates_dic={p[0].attribute: p for p in filtered_predicates},
        )

        return causal_model

    def compute_confidence(
        self,
        causal_model: CausalModel,
        data: AnomalyData,
    ) -> Tuple[float, float]:
        """Compute the confidence of the causal model"""
        # Create partitions
        partitions_by_attr: List[List[Partition]] = self.create_partitions(data)
        # Label partitions
        partitions_labeled: List[List[Partition]] = []
        for idx, partitions in enumerate(partitions_by_attr):
            labeled_partitions: List[Partition] = self.label_parition(
                values=data.valid_values_as_np[:, idx],
                partitions=partitions,
                normal_regions=data.valid_normal_regions,
                abnormal_regions=data.valid_abnormal_regions,
            )
            partitions_labeled.append(labeled_partitions)

        # Get only the partitions to be used for extracting predicates
        partitions_to_use: List[List[Partition]] = list(
            filter(self.is_to_extract_predicates, partitions_labeled)
        )

        total_confidence = 0
        total_precision = 0
        # TODO: average를 predicates 기준으로 해야함. partition에 대해서는 더 해줘야하는데, matlab 코드에서 line:532 참조 필요
        for partitions in partitions_to_use:
            # Count the number of normal and abnormal partitions satisfying the predicate of causal model
            num_normal_partitions = 0
            num_abnormal_partitions = 0
            num_valid_normal_partitions = 0
            num_valid_abnormal_partitions = 0
            # Count
            for partition in partitions:
                if partition.is_normal:
                    num_normal_partitions += 1
                    if causal_model.is_valid_partition(partition):
                        num_valid_normal_partitions += 1
                elif partition.is_abnormal:
                    num_abnormal_partitions += 1
                    if causal_model.is_valid_partition(partition):
                        num_valid_abnormal_partitions += 1

            # Compute confidence
            covered_normal_ratio = num_valid_normal_partitions / num_normal_partitions
            covered_abnormal_ratio = (
                num_valid_abnormal_partitions / num_abnormal_partitions
            )
            confidence = covered_abnormal_ratio - covered_normal_ratio
            precision = covered_abnormal_ratio / (
                covered_abnormal_ratio + covered_normal_ratio
            )
            # Aggregate
            total_confidence += confidence
            total_precision += precision
        avg_total_precision = total_precision / len(partitions_to_use) * 100
        avg_total_confidence = total_confidence / len(partitions_to_use) * 100
        return avg_total_precision, avg_total_confidence
