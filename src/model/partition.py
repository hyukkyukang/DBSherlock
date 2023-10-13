from typing import *

import hkkang_utils.data as data_utils
import hkkang_utils.pattern as pattern_utils


@data_utils.dataclass
class Label(pattern_utils.SingletonABCMeta):
    pass


@data_utils.dataclass
class Normal(Label):
    pass


@data_utils.dataclass
class Abnormal(Label):
    pass


@data_utils.dataclass
class Empty(Label):
    pass


@data_utils.dataclass
class Partition:
    max: float
    min: float
    label: Optional[Label]

    # Syntactic sugars
    @property
    def size(self) -> float:
        return self.max - self.min

    @property
    def is_empty(self) -> bool:
        return self.label == Empty()

    @property
    def is_normal(self) -> bool:
        return self.label == Normal()

    @property
    def is_abnormal(self) -> bool:
        return self.label == Abnormal()

    @is_empty.setter
    def is_empty(self, value: bool) -> None:
        if value:
            self.label = Empty()
        else:
            raise ValueError("Cannot set is_empty to False")

    @is_normal.setter
    def is_normal(self, value: bool) -> None:
        if value:
            self.label = Normal()
        else:
            raise ValueError("Cannot set is_normal to False")

    @is_abnormal.setter
    def is_abnormal(self, value: bool) -> None:
        if value:
            self.label = Abnormal()
        else:
            raise ValueError("Cannot set is_abnormal to False")

    # Helper functions
    def is_value_in_range(self, value: float) -> bool:
        return self.min <= value and value < self.max
