from typing import *

import hkkang_utils.data as data_utils

from src.model.partition import Partition


@data_utils.dataclass
class Predicate:
    attribute: str
    operand1: float
    operator1: str
    operand2: Optional[float] = data_utils.field(default=None)
    operator2: Optional[str] = data_utils.field(default=None)

    @property
    def is_unary(self) -> bool:
        return self.operand2 is None and self.operator2 is None

    @property
    def is_binary(self) -> bool:
        return not self.is_unary

    def __repr__(self):
        if self.is_unary:
            return f"{self.attribute} {self.operator1} {self.operand1}"
        else:
            return f"{self.operand1} {self.operator2} {self.attribute} {self.operator2} {self.operand2}"


@data_utils.dataclass
class CausalModel:
    cause: str
    predicates_dic: Dict[str, List[Predicate]]  # List of effective predicates

    def _do_satisfy_predicate(self, predicate: Predicate, partition: Partition) -> bool:
        """Check if the partition satisfies the predicate"""
        if predicate.is_unary:
            if predicate.operator1 == ">":
                return partition.min >= predicate.operand1
            else:
                return partition.max <= predicate.operand1
        else:
            assert predicate.operator1 == ">" and predicate.operator2 == "<"
            return (
                partition.min >= predicate.operand1
                and partition.max <= predicate.operand2
            )

    def is_valid_partition(self, partition: Partition) -> bool:
        """Check if the partition satisfies any of the effective predicates"""
        if partition.attribute not in self.predicates_dic:
            return False
        return any(
            self._do_satisfy_predicate(predicate, partition)
            for predicate in self.predicates_dic[partition.attribute]
        )
