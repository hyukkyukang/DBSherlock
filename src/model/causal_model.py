import copy
from typing import *

import hkkang_utils.data as data_utils

from src.model.partition import Partition


@data_utils.dataclass
class Predicate:
    """For operators, assume that the variables are on the left side"""

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

    def __add__(self, other: "Predicate") -> "Predicate":
        # Check attribute
        assert (
            self.attribute == other.attribute
        ), f"Addition only supported for the same predicates. But found: {self.attribute} vs {other.attribute}"
        # Check operators
        assert self.operator1 == other.operator1, f"Invalid predicate: {self} + {other}"
        assert (
            self.operator2 == other.operator2
        ), f"Invalide predicate: {self} + {other}"
        assert self.operator1 == ">", f"Invalid operator1: {self.operator1}"
        assert self.operator2 in ["<", None], f"Invalid operator2: {self.operator2}"
        # Condition 1: if both are unary
        if self.is_unary and other.is_unary:
            new_predicate = Predicate(
                attribute=self.attribute,
                operand1=min(self.operand1, other.operand1),
                operator1=">",
            )
        elif self.is_unary and other.is_binary:
            # The current predicate covers the other predicate
            if self.operand1 >= other.operand2:
                new_predicate = Predicate(
                    attribute=self.attribute,
                    operand1=self.operand1,
                    operator1=">",
                )
            else:
                new_predicate = Predicate(
                    attribute=self.attribute,
                    operand1=min(self.operand1, other.operand1),
                    operator1=">",
                    operand2=other.operand2,
                    operator2="<",
                )
        elif self.is_binary and other.is_unary:
            # The other predicate covers the current predicate
            if other.operand1 >= self.operand2:
                new_predicate = Predicate(
                    attribute=self.attribute,
                    operand1=other.operand1,
                    operator1=">",
                )
            else:
                new_predicate = Predicate(
                    attribute=self.attribute,
                    operand1=min(self.operand1, other.operand1),
                    operator1=">",
                    operand2=self.operand2,
                    operator2="<",
                )
        elif self.is_binary and other.is_binary:
            new_predicate = Predicate(
                attribute=self.attribute,
                operand1=min(self.operand1, other.operand1),
                operator1=">",
                operand2=max(self.operand2, other.operand2),
                operator2="<",
            )

        # Return new instance
        return new_predicate


@data_utils.dataclass
class CausalModel:
    cause: str
    predicates_dic: Dict[str, Predicate]  # Effective predicates

    def _do_satisfy_predicate(self, predicate: Predicate, partition: Partition) -> bool:
        """Check if the partition satisfies the predicate"""
        if predicate.is_unary:
            if predicate.operator1 == ">":
                return partition.min >= predicate.operand1
            else:
                return partition.max <= predicate.operand1
        else:
            assert (
                predicate.operator1 == ">" and predicate.operator2 == "<"
            ), f"Invalid predicate: {predicate}"
            return (
                partition.min >= predicate.operand1
                and partition.max <= predicate.operand2
            )

    def is_valid_partition(self, partition: Partition) -> bool:
        """Check if the partition satisfies any of the effective predicates"""
        if partition.attribute not in self.predicates_dic:
            return False
        return self._do_satisfy_predicate(
            self.predicates_dic[partition.attribute], partition
        )

    def __add__(self, other: "CausalModel") -> "CausalModel":
        """Not in-place addition"""
        assert (
            self.cause == other.cause
        ), f"Addition is supported on the same anomaly causes. But found: {self.cause} vs {other.cause}"
        predicate_dic1 = self.predicates_dic
        predicate_dic2 = other.predicates_dic
        new_predicate_dic = {}
        # Select common attributes
        common_attributes = set(predicate_dic1.keys()) & set(predicate_dic2.keys())
        for attribute in common_attributes:
            predicate1 = predicate_dic1[attribute]
            predicate2 = predicate_dic2[attribute]
            # Union of predicates
            new_predicate_dic[attribute] = predicate1 + predicate2
        # Return new instance
        return CausalModel(
            cause=self.cause,
            predicates_dic=new_predicate_dic,
        )

    def __radd__(self, other) -> "CausalModel":
        if type(other) == int and other == 0:
            return copy.deepcopy(self)
        return self.__add__(other)
